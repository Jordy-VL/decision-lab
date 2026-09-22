"""One CE epoch and development/calibration evaluation, no test access.

Run: uv run modal run --detach scripts/modal_baseline.py
Artifacts persist in the decision-lab-profile volume even if the client disconnects.
"""
from pathlib import Path
import json
import subprocess
from datetime import datetime, timezone
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("decision-lab-ce-baseline")
volume = modal.Volume.from_name("decision-lab-profile", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.57.6", "PyYAML==6.0.3")
         .env({"HF_HOME": "/artifacts/hf", "HF_HUB_OFFLINE": "1",
               "PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(ROOT / "packages/modernbert-decisions/decisions", "/workspace/decisions")
         .add_local_file(ROOT / "packages/modernbert-decisions/configs/ce-baseline.yaml", "/workspace/ce.yaml"))
for split in ("train", "development", "calibration"):
    image = image.add_local_file(ROOT / f"data/kev-decision-v7/{split}.jsonl", f"/workspace/{split}.jsonl")


@app.function(image=image, gpu="A100", cpu=2, memory=16384,
              volumes={"/artifacts": volume}, timeout=4500, retries=0,
              max_containers=1, scaledown_window=2)
def run_baseline(run_id, code_revision):
    import time
    import sys
    import traceback
    import torch
    started = time.perf_counter()
    root = Path("/artifacts/runs") / run_id
    root.mkdir(parents=True, exist_ok=False)
    status = {"run_id": run_id, "code_revision": code_revision,
              "status": "running", "gpu": torch.cuda.get_device_name(),
              "precision": "float32; no autocast", "timeout_seconds": 4500,
              "budget_note": "User total first-attempt cap USD 10; this invocation limited to 75 GPU minutes, no retries."}

    def persist():
        status["elapsed_seconds"] = time.perf_counter() - started
        (root / "status.json").write_text(json.dumps(status, indent=2))
        volume.commit()

    def execute(args, log_name, timeout):
        with (root / log_name).open("w") as log:
            # Output is retained in the volume and streamed to Modal logs.
            proc = subprocess.Popen([sys.executable, "-m", "decisions", *args],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            import threading
            timer = threading.Timer(timeout, proc.kill)
            timer.start()
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if proc.wait() != 0:
                    raise RuntimeError(f"{log_name} failed or exceeded its time budget")
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    persist()
    try:
        execute(["train", "--config", "/workspace/ce.yaml", "--data", "/workspace/train.jsonl",
                 "--output", str(root / "train"), "--device", "cuda"], "train.log", 3300)
        status["status"] = "trained"
        persist()
        for split in ("development", "calibration"):
            execute(["evaluate", "--config", "/workspace/ce.yaml",
                     "--checkpoint", str(root / "train/checkpoint"),
                     "--data", f"/workspace/{split}.jsonl", "--split", split,
                     "--output", str(root / split), "--device", "cuda"], f"{split}.log", 450)
            status[split] = json.loads((root / split / "report.json").read_text())
            persist()
        status["status"] = "complete"
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        persist()
    return status


@app.local_entrypoint()
def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_id = "ce-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps({"run_id": run_id, "code_revision": revision,
                                                 "volume": "decision-lab-profile"}, indent=2))
    print(f"Run: {run_id}", flush=True)
    result = run_baseline.remote(run_id, revision)
    (output / "status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
