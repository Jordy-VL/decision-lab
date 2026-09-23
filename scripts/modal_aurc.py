"""Run one AURC continuation from the frozen CE checkpoint on Modal.

The function trains one matched-budget continuation and evaluates only the
development and calibration partitions. It never downloads or reads test data.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("decision-lab-aurc-continuation")
volume = modal.Volume.from_name("decision-lab-profile", create_if_missing=False)
config_path = ROOT / "packages/modernbert-decisions/configs/continue-aurc.yaml"
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.57.6", "PyYAML==6.0.3")
         .env({"HF_HOME": "/artifacts/hf", "HF_HUB_OFFLINE": "1",
               "PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(ROOT / "packages/modernbert-decisions/decisions", "/workspace/decisions")
         .add_local_file(config_path, "/workspace/continue-aurc.yaml"))
for split in ("train", "development", "calibration"):
    image = image.add_local_file(ROOT / f"data/kev-decision-v7/{split}.jsonl", f"/workspace/{split}.jsonl")


@app.function(image=image, gpu="A100-40GB", cpu=2, memory=16384,
              volumes={"/artifacts": volume}, timeout=4500, retries=0,
              max_containers=1, scaledown_window=2)
def run_aurc(run_id: str, code_revision: str, parent_run_id: str):
    import shutil
    import sys
    import time
    import traceback
    import torch

    started = time.perf_counter()
    root = Path("/artifacts/runs") / run_id
    root.mkdir(parents=True, exist_ok=False)
    parent = Path("/artifacts/runs") / parent_run_id / "train/checkpoint"
    if not (parent / "model.json").is_file() or not (parent / "head.pt").is_file():
        raise FileNotFoundError(f"frozen parent checkpoint is incomplete: {parent}")
    status = {"run_id": run_id, "parent_run_id": parent_run_id,
              "code_revision": code_revision, "status": "running",
              "objective": "CE + 0.5 * detached harmonic rank-weighted CE",
              "gpu": torch.cuda.get_device_name(), "precision": "float32; no autocast",
              "timeout_seconds": 4500, "retries": 0,
              "budget_note": "One continuation only; development/calibration only; no test access."}

    def persist():
        status["elapsed_seconds"] = time.perf_counter() - started
        (root / "status.json").write_text(json.dumps(status, indent=2))
        volume.commit()

    def execute(args, log_name, timeout):
        with (root / log_name).open("w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "decisions", *args],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd="/workspace")
            import threading
            timer = threading.Timer(timeout, proc.kill)
            timer.start()
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = {}
                    if "recovery_checkpoint" in event:
                        status["latest_recovery"] = event["recovery_checkpoint"]
                        status["optimizer_steps"] = event["optimizer_steps"]
                        persist()
                        recovery_root = root / "train/recovery"
                        checkpoints = sorted(p for p in recovery_root.glob("step-*")
                                             if (p / "COMPLETE").is_file())
                        for old in checkpoints[:-2]:
                            if old.parent.resolve() != recovery_root.resolve():
                                raise ValueError("invalid recovery checkpoint path")
                            shutil.rmtree(old)
                        volume.commit()
                if proc.wait() != 0:
                    raise RuntimeError(f"{log_name} failed or exceeded its time budget")
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    persist()
    try:
        common = ["--config", "/workspace/continue-aurc.yaml",
                  "--data-revision", "kev-decision-v7-a88f56db-adapter-v1",
                  "--checkpoint", str(parent), "--device", "cuda",
                  "--seed", "17", "--batch-size", "4", "--accumulation", "1",
                  "--epochs", "1", "--learning-rate", "0.00002",
                  "--weight-decay", "0.01", "--aurc-lambda", "0.5",
                  "--save-every", "250", "--gradient-checkpointing"]
        execute(["train", *common, "--data", "/workspace/train.jsonl",
                 "--output", str(root / "train")], "train.log", 3600)
        status["training"] = json.loads((root / "train/training.json").read_text())
        status["status"] = "trained"
        persist()
        for split in ("development", "calibration"):
            execute(["evaluate", *common, "--data", f"/workspace/{split}.jsonl",
                     "--split", split, "--output", str(root / split)],
                    f"{split}.log", 450)
            status[split] = json.loads((root / split / "report.json").read_text())
            persist()
        status["status"] = "complete"
    except BaseException:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        persist()
    return status


@app.local_entrypoint()
def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    parent_run_id = "ce-20260923-080034"
    run_id = "aurc-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps({
        "run_id": run_id, "parent_run_id": parent_run_id,
        "code_revision": revision, "volume": "decision-lab-profile"}, indent=2))
    call = run_aurc.spawn(run_id, revision, parent_run_id)
    (output / "call.json").write_text(json.dumps({
        "function_call_id": call.object_id, "run_id": run_id}, indent=2))
    print(f"Submitted {run_id}: {call.object_id}", flush=True)
