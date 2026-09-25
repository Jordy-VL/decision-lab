"""End-to-end CPU smoke for Trainer, resume, best/final checkpoints and logits."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml
from transformers import BertConfig, BertModel, BertTokenizerFast, TrainerCallback

from decisions.cli import main as decisions_main, predict
from decisions.config import Config
from decisions.data import load
from decisions.hf_trainer import create_trainer
from decisions.model import DecisionModel, encode, initialize


class StopAfterStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def create_fixture(root):
    model_dir = root / "tiny-bert"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "state", "question", "options",
             "signal", "positive", "negative", "choose", "the", "matching", "answer", "0", "1", "2"]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(20260923)
    encoder = BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, type_vocab_size=2, pad_token_id=0,
    ))
    encoder.save_pretrained(model_dir)

    def row(split, i, positive):
        return {
            "id": f"{split}-{i}", "group_id": f"{split}-group-{i}", "source": "synthetic-smoke",
            "split": split, "state": f"signal {'positive' if positive else 'negative'}",
            "question": "choose the matching answer", "options": ["negative", "positive"],
            "target_index": 1 if positive else 0, "type": "categorical",
        }

    paths = {}
    for split, positives in (("train", [True, True, True, True]),
                             ("development", [False, False]), ("calibration", [True, False])):
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row(split, i, value)) + "\n" for i, value in enumerate(positives)),
                        encoding="utf-8")
        paths[split] = path
    return model_dir, paths


def assert_numpy_artifacts(run_dir, split, source_path, config, device):
    data = run_dir / "evaluations" / "best" / split
    with np.load(data / "logits.npz", allow_pickle=False) as saved:
        logits = saved["logits"]
        labels = saved["labels"]
        ids = saved["ids"].tolist()
        counts = saved["option_counts"].tolist()
    rows = [json.loads(line) for line in (data / "predictions.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in source_path.read_text().splitlines()]
    check(ids == [row["id"] for row in expected] == [row["id"] for row in rows], f"{split}: ID order mismatch")
    check(labels.tolist() == [row["target_index"] for row in expected], f"{split}: labels mismatch")
    for i, row in enumerate(rows):
        n = counts[i]
        check(n == len(row["options"]), f"{split}: option count mismatch at {row['id']}")
        check(np.isfinite(logits[i, :n]).all(), f"{split}: nonfinite active logits at {row['id']}")
        check(np.isneginf(logits[i, n:]).all(), f"{split}: padded logits are not -inf at {row['id']}")
        check(np.allclose(logits[i, :n], row["logits"], rtol=0, atol=1e-7),
              f"{split}: NPZ and JSONL logits differ at {row['id']}")

    best, best_tokenizer = DecisionModel.load(run_dir / "checkpoint")
    best.to(device)
    examples = [encode(row, best_tokenizer, config.max_length, config.max_options) for row in expected]
    model_rows = predict(best, best_tokenizer, examples, config, device)
    for saved_row, model_row in zip(rows, model_rows):
        check(np.allclose(saved_row["logits"], model_row["logits"], rtol=0, atol=1e-6),
              f"{split}: best checkpoint logits differ at {saved_row['id']}")


def load_training_examples(config, paths, model_dir):
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    train_config = replace(config, data=str(paths["train"]), data_split="")
    dev_config = replace(config, data=str(paths["development"]), data_split="")
    calibration_config = replace(config, data=str(paths["calibration"]), data_split="")
    return tokenizer, [encode(row, tokenizer, config.max_length, config.max_options)
                       for row in load(paths["train"], train_config)], \
        [encode(row, tokenizer, config.max_length, config.max_options)
         for row in load(paths["development"], dev_config)], \
        [encode(row, tokenizer, config.max_length, config.max_options)
         for row in load(paths["calibration"], calibration_config)]


def compare_final_models(first, second):
    model_a, _ = DecisionModel.load(first)
    model_b, _ = DecisionModel.load(second)
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    check(state_a.keys() == state_b.keys(), "resume changed model parameter keys")
    for key in state_a:
        check(torch.equal(state_a[key], state_b[key]), f"exact resume diverged at parameter {key}")


def upload_smoke_artifacts(run_dir, repo_id):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    destination = datetime.now(timezone.utc).strftime("smoke/%Y%m%dT%H%M%SZ")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=run_dir,
        path_in_repo=destination,
        commit_message=f"Upload Trainer smoke artifacts {destination.rsplit('/', 1)[-1]}",
    )
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-batch", action="store_true", help="run one optimizer update and artifact check only")
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu",
                        help="use CPU locally or let Trainer use the available accelerator")
    parser.add_argument("--require-cuda", action="store_true", help="fail unless CUDA is available")
    parser.add_argument("--keep-output", type=Path, help="keep smoke fixture/artifacts in this new or empty directory")
    parser.add_argument("--upload-hf", action="store_true",
                        help="upload the completed smoke run to a private Hugging Face model repository")
    parser.add_argument("--hf-repo-id", default="jordyvl/decision-lab-smoke",
                        help="model repository for --upload-hf")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo / "runs"
    runs.mkdir(exist_ok=True)
    temp = None
    if args.keep_output:
        root = args.keep_output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        check(not any(root.iterdir()), f"--keep-output directory must be empty: {root}")
    else:
        temp = tempfile.TemporaryDirectory(prefix="hf-trainer-smoke-", dir=runs)
        root = Path(temp.name)
    try:
        torch.set_num_threads(1)
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("--require-cuda was set but CUDA is unavailable")
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if args.device == "auto" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        model_dir, paths = create_fixture(root)
        if args.single_batch:
            single_batch_train = root / "single-batch-train.jsonl"
            train_lines = paths["train"].read_text(encoding="utf-8").splitlines()
            single_batch_train.write_text("\n".join(train_lines[:2]) + "\n", encoding="utf-8")
            paths["train"] = single_batch_train
        cli_run = root / "cli-run"
        config = Config(
            model=str(model_dir), revision="main", data=str(paths["train"]),
            development_data=str(paths["development"]), calibration_data=str(paths["calibration"]),
            output=str(cli_run), save_every=1, logging_steps=1, eval_accumulation_steps=1,
            seed=23, max_length=64, head_hidden=16, max_options=4,
            batch_size=2, accumulation=1, epochs=1 if args.single_batch else 2, learning_rate=0.001,
            weight_decay=0, warmup_ratio=0, gradient_checkpointing=False, device=args.device,
        )
        config_path = root / "smoke-config.yaml"
        config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
        decisions_main(["train", "--config", str(config_path)])

        state = json.loads((cli_run / "trainer_state.json").read_text())
        history = json.loads((cli_run / "trainer_log_history.json").read_text())
        evaluated = [entry for entry in history if "eval_augrc" in entry]
        check(evaluated, "Trainer log history contains no development AUGRC evaluations")
        selected_eval = min(evaluated, key=lambda entry: entry["eval_augrc"])
        check(Path(state["best_model_checkpoint"]).name == f"checkpoint-{selected_eval['step']}",
              "Trainer best checkpoint does not match minimum development AUGRC")
        if not args.single_batch:
            check(selected_eval["step"] < state["max_steps"],
                  "smoke fixture did not exercise restoration from a non-final best checkpoint")
        best_checkpoint = Path(state["best_model_checkpoint"])
        check(best_checkpoint.is_dir(), "Trainer best checkpoint directory is missing")
        check((cli_run / "checkpoint" / "model.json").exists(), "project best checkpoint is missing")
        final_dir = cli_run / "final_budget_checkpoint"
        check((final_dir / "model.json").exists(), "final-budget model is missing")
        final_step_checkpoint = cli_run / "trainer" / f"checkpoint-{state['max_steps']}"
        check(final_step_checkpoint.is_dir(), "final HF Trainer checkpoint is missing")
        final_model, _ = DecisionModel.load(final_dir)
        hf_final_state = torch.load(final_step_checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
        check(final_model.state_dict().keys() == hf_final_state.keys(), "final-budget weights differ in structure from final HF checkpoint")
        for key, value in final_model.state_dict().items():
            check(torch.equal(value, hf_final_state[key]), f"final-budget weights do not match final HF step at {key}")
        for split in ("development", "calibration"):
            assert_numpy_artifacts(cli_run, split, paths[split], config, device)

        resume_checkpoint = None
        if not args.single_batch:
            # Compare uninterrupted training to a stop-at-step-2 exact Trainer resume.
            tokenizer, train_rows, dev_rows, _ = load_training_examples(config, paths, model_dir)
            uninterrupted_dir, resumed_dir = root / "uninterrupted", root / "resumed"
            torch.manual_seed(config.seed)
            model, _ = initialize(config)
            uninterrupted = create_trainer(model, tokenizer, train_rows, dev_rows, config, uninterrupted_dir)
            uninterrupted.train()

            torch.manual_seed(config.seed)
            model, _ = initialize(config)
            interrupted = create_trainer(model, tokenizer, train_rows, dev_rows, config, resumed_dir)
            interrupted.add_callback(StopAfterStep(2))
            interrupted.train()
            resume_checkpoint = resumed_dir / "trainer" / "checkpoint-2"
            check(resume_checkpoint.is_dir(), "interrupted run did not save checkpoint-2")

            torch.manual_seed(config.seed)
            model, _ = initialize(config)
            resumed = create_trainer(model, tokenizer, train_rows, dev_rows, config, resumed_dir)
            resumed.train(resume_from_checkpoint=str(resume_checkpoint))
            check(resumed.state.global_step == uninterrupted.state.global_step == 4,
                  "resumed Trainer did not finish at the configured optimizer step")
            compare_final_models(uninterrupted_dir / "final_budget_checkpoint", resumed_dir / "final_budget_checkpoint")
        result_device = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        upload_url = None
        if args.upload_hf:
            upload_url = str(upload_smoke_artifacts(cli_run, args.hf_repo_id))
        print(json.dumps({"result": "PASS", "device": result_device, "training_steps": state["max_steps"],
                          "best_checkpoint": str(best_checkpoint),
                          "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                          "logit_splits": ["development", "calibration"],
                          "huggingface_commit": upload_url,
                          "artifacts": str(root) if args.keep_output else "temporary artifacts cleaned"}, indent=2))
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
