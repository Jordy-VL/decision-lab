"""Trusted local recovery bundles, atomically published at optimizer boundaries."""
import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict
import torch


def signature(config):
    values = asdict(config)
    for key in ("output", "checkpoint", "resume", "save_every", "device"):
        values.pop(key)
    values["data"] = hashlib.sha256(Path(config.data).read_bytes()).hexdigest()
    return values


def save(model, tokenizer, optimizer, config, progress):
    root = Path(config.output) / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"step-{progress['updates']:06d}"
    temp = root / (dest.name + ".incomplete")
    temp.mkdir(exist_ok=False)
    model.save(temp, tokenizer)
    torch.save(dict(optimizer=optimizer.state_dict(), progress=progress,
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    signature=signature(config)), temp / "training_state.pt")
    (temp / "COMPLETE").write_text("complete\n")
    os.rename(temp, dest)
    pointer = root / "latest.tmp"
    pointer.write_text(json.dumps({"checkpoint": dest.name, "updates": progress["updates"]}))
    os.replace(pointer, root / "latest.json")
    print(json.dumps({"recovery_checkpoint": str(dest), "optimizer_steps": progress["updates"]}), flush=True)


def restore(path, optimizer, config):
    path = Path(path)
    if not (path / "COMPLETE").exists():
        raise ValueError("resume requires a completed recovery bundle")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state["signature"] != signature(config):
        raise ValueError("resume configuration/data differs from saved run")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]
