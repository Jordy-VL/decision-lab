"""Bounded CE profiling only: uv run modal run scripts/modal_profile.py."""
from pathlib import Path
import json
import subprocess
import modal

ROOT = Path(__file__).resolve().parents[1]
REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
app = modal.App("decision-lab-profile")
volume = modal.Volume.from_name("decision-lab-profile", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.57.6", "PyYAML==6.0.3")
         .env({"HF_HOME": "/artifacts/hf", "PYTHONPATH": "/workspace"})
         .add_local_dir(ROOT / "packages/modernbert-decisions/decisions", "/workspace/decisions")
         .add_local_file(ROOT / "data/kev-decision-v7/train.jsonl", "/workspace/train.jsonl"))


@app.function(image=image, volumes={"/artifacts": volume}, timeout=600, retries=0)
def prepare():
    from huggingface_hub import snapshot_download
    snapshot_download("answerdotai/ModernBERT-large", revision=REVISION,
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
    volume.commit()


@app.function(image=image, gpu="A100", cpu=2, memory=16384,
              volumes={"/artifacts": volume}, timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def profile(code_revision):
    import time
    import random
    import hashlib
    import statistics
    import importlib.metadata
    import torch
    from decisions.config import Config
    from decisions.data import load
    from decisions.model import initialize, encode, collate
    from decisions.loss import decision_loss
    started = time.perf_counter()
    torch.manual_seed(17)
    random.seed(17)
    cfg = Config(revision=REVISION, data="/workspace/train.jsonl", device="cuda",
                 gradient_checkpointing=True)
    rows = load(cfg.data, cfg, labeled=True)
    model, tokenizer = initialize(cfg)
    examples = [encode(r, tokenizer, cfg.max_length, cfg.max_options) for r in rows]
    random.Random(17).shuffle(examples)
    model.to("cuda").train()
    model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    timings, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(32):
        items = examples[step * 4:step * 4 + 4]
        batch = {k: v.cuda() for k, v in collate(items, tokenizer.pad_token_id, 128).items()}
        targets = torch.tensor([e["row"]["target_index"] for e in items], device="cuda")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = decision_loss(model(**batch).logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - tick)
        losses.append(loss.item())
        print(json.dumps({"step": step + 1, "seconds": timings[-1], "loss": losses[-1]}), flush=True)
    # Check a conservative padding case separately from the random throughput sample.
    longest = sorted(examples, key=lambda e: len(e["input_ids"]), reverse=True)[:4]
    optimizer.zero_grad(set_to_none=True)
    batch = {k: v.cuda() for k, v in collate(longest, tokenizer.pad_token_id, 128).items()}
    loss = decision_loss(model(**batch).logits, torch.tensor([e["row"]["target_index"] for e in longest], device="cuda"))
    loss.backward()
    torch.cuda.synchronize()
    steady = statistics.mean(timings[4:])
    report = {"code_revision": code_revision, "model_revision": REVISION,
              "data_sha256": hashlib.sha256(Path(cfg.data).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "precision": "float32; no autocast",
              "batch_size": 4, "gradient_checkpointing": True,
              "train_rows": len(rows), "profile_updates": 32,
              "step_seconds": timings, "losses": losses,
              "steady_step_seconds": steady, "examples_per_second": 4 / steady,
              "projected_epoch_seconds": ((len(rows) + 3) // 4) * steady,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "longest_batch_tokens": [len(e["input_ids"]) for e in longest],
              "gpu_function_seconds": time.perf_counter() - started,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "PyYAML")},
              "note": "Timing pilot only; no checkpoint or quality evaluation. Epoch projection excludes checkpoint/evaluation overhead."}
    Path("/artifacts/profile.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prepare.remote()
    report = profile.remote(revision)
    output = ROOT / "runs/modal-profile"
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
