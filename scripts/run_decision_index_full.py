"""Build, verify, smoke, run and score Decision Index; resume only identical inputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT_REVISION = "87d4650b42b377c0291a89c1f1a879f9b31082bf"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--kit", type=Path, default=Path("runs/decision-index-kit-20260930"))
    parser.add_argument("--work", type=Path, default=Path("runs/decision-index-work-20260930"))
    parser.add_argument("--suite", type=Path, default=Path("runs/decision-index-suite-0.2.1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        parser.error("run-name must be a simple filename")
    os.chdir(ROOT)
    checkpoint = args.checkpoint.resolve()
    if json.loads((checkpoint / "model.json").read_text())["decision_head"] != "fixed_slot":
        parser.error("this job is for the requested fixed-slot checkpoint")
    kit = args.kit.resolve()
    revision = subprocess.check_output(["git", "-C", str(kit), "rev-parse", "HEAD"], text=True).strip()
    if revision != KIT_REVISION or subprocess.check_output(["git", "-C", str(kit), "status", "--porcelain"], text=True).strip():
        parser.error("kit must be the clean pinned revision")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", gpu):
        parser.error("set CUDA_VISIBLE_DEVICES to one authorized GPU")
    out = ROOT / "runs" / ("decision-index-job-" + args.run_name)
    if out.exists() and not args.resume:
        parser.error("job already exists; use --resume for the same frozen inputs")
    out.mkdir(exist_ok=True)
    lock = (out / "job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, **extra):
        data = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **extra)
        temp = out / "job-status.tmp"
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(out / "job-status.json")
        print(json.dumps(data), flush=True)

    env = dict(os.environ, HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "packages/modernbert-decisions"), str(kit)])
    # The builder fetches and hash-verifies ESCI LFS payloads itself. The server
    # lacks git-lfs, so leave pointers during Git checkout, scoped to this job.
    env.update(GIT_CONFIG_COUNT="3", GIT_CONFIG_KEY_0="filter.lfs.process", GIT_CONFIG_VALUE_0="",
               GIT_CONFIG_KEY_1="filter.lfs.smudge", GIT_CONFIG_VALUE_1="cat",
               GIT_CONFIG_KEY_2="filter.lfs.required", GIT_CONFIG_VALUE_2="false")

    def run(phase, argv):
        status(phase)
        subprocess.run([sys.executable, "-u", "-m", "decision_index", *argv], env=env, check=True)

    try:
        status("pinning_inputs")
        source_paths = [ROOT / "scripts/jev_decision_index_engine.py", Path(__file__),
                        *sorted((ROOT / "packages/modernbert-decisions/decisions").glob("*.py"))]
        identity = dict(checkpoint=str(checkpoint), edition="0.2.1", kit_revision=revision,
                        checkpoint_sha256={str(p.relative_to(checkpoint)): digest(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
                        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                        suite=str(args.suite.resolve()), work=str(args.work.resolve()),
                        max_length=2048, question_batch_size=8)
        manifest = out / "provenance.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("refusing resume with changed checkpoint, source or configuration")
        else:
            manifest.write_text(json.dumps(identity, indent=2) + "\n")
        sys.path.insert(0, str(kit))
        from decision_index.suite.io import Suite
        try:
            suite = Suite(args.suite, "0.2.1")
        except FileNotFoundError:
            if shutil.disk_usage(ROOT / "runs").free < 25 * 1024**3:
                raise RuntimeError("suite rebuild requires at least 25 GiB available")
            run("building_suite", ["suite", "rebuild", "--edition", "0.2.1", "--work", str(args.work)])
            rebuilt = args.work / "artifacts/benchmark-suite/release-v2-rebuilt"
            run("importing_suite", ["suite", "import", "--edition", "0.2.1", "--dir", str(args.suite),
                                    "--rows", str(rebuilt / "selected-rows.jsonl.gz"),
                                    "--added-rows", str(rebuilt / "added-rows.jsonl.gz")])
            suite = Suite(args.suite, "0.2.1")
        verified = suite.verify(strict=True)
        if not verified.get("exclusions_match"):
            raise ValueError("missing or mismatched official exclusion list")
        (out / "suite-verification.json").write_text(json.dumps(verified, indent=2) + "\n")
        usage = subprocess.check_output(["nvidia-smi", "--id=" + gpu,
                 "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if any(int(v.strip()) > 10 for v in usage.strip().split(",")):
            raise RuntimeError("authorized GPU is busy; resume when it is free")
        common = ["--edition", "0.2.1", "--suite-dir", str(args.suite),
                  "--engine", "jev_decision_index_engine:ModernBERTDecisionEngine",
                  "--option", "checkpoint=" + str(checkpoint), "--option", "device=cuda",
                  "--option", "max_length=2048", "--option", "question_batch_size=8", "--compact"]
        sample = out / "sample-rows.jsonl.gz"
        if not sample.exists():
            run("sampling", ["suite", "sample", "--edition", "0.2.1", "--dir", str(args.suite),
                             "--n", "20", "--out", str(sample)])
        run("sample_inference_and_scoring", ["pipeline", *common, "--rows", str(sample), "--out", str(out / "sample")])
        sample_scores = json.loads((out / "sample/scores.json").read_text())
        if sample_scores["counts"].get("error", 0) or not sample_scores["counts"].get("ok", 0):
            raise RuntimeError("sample did not pass: inspect unsupported/error counts")
        run("full_inference_and_scoring", ["pipeline", *common, "--out", str(out / "full")])
        scores = json.loads((out / "full/scores.json").read_text())
        if not scores["complete"] or scores["counts"].get("error", 0):
            raise RuntimeError("full run has missing or errored requests; inspect before resuming")
        status("completed", completed=scores["completed"], counts=scores["counts"], decision_index=scores["decision_index"])
    except Exception as error:
        status("failed", exception=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
