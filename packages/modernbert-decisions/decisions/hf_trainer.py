"""Hugging Face Trainer adapter for the typed decision model and AURC surrogate."""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from .loss import decision_loss
from .model import collate


class EncodedRows(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class FinalBudgetCheckpointCallback(TrainerCallback):
    """Keep a project-format copy of the final-step model before Trainer loads the best model."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.saved = False

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if not self.saved and state.global_step >= state.max_steps and args.should_save:
            model.save(self.output_dir, processing_class)
            self.saved = True
        return control


def make_collator(tokenizer, config):
    def collate_rows(examples):
        batch = collate(examples, tokenizer.pad_token_id, config.max_options)
        batch["labels"] = torch.tensor([e["row"]["target_index"] for e in examples], dtype=torch.long)
        batch["task_types"] = [e["row"]["type"] for e in examples]
        return batch
    return collate_rows


class DecisionTrainer(Trainer):
    """Use Trainer's optimizer, scheduler, checkpoints, logging, evaluation and resume."""
    model_accepts_loss_kwargs = False

    def __init__(self, *args, aurc_lambda=0.0, rank_by_type=False, final_budget_dir=None, **kwargs):
        self.aurc_lambda = aurc_lambda
        self.rank_by_type = rank_by_type
        self.final_budget_dir = Path(final_budget_dir) if final_budget_dir else None
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        task_types = inputs.pop("task_types", None)
        if not self.rank_by_type:
            task_types = None
        logits = model(**inputs).logits
        loss = decision_loss(logits, labels, self.aurc_lambda, task_types)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training/evaluation loss")
        return (loss, {"logits": logits}) if return_outputs else loss

def make_training_arguments(config, output_dir, cadence):
    """Resolve the experiment config into standard HF TrainingArguments."""
    return TrainingArguments(
        output_dir=str(Path(output_dir) / "trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.accumulation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=cadence,
        save_strategy="steps",
        save_steps=cadence,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="nll",
        greater_is_better=False,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        bf16=config.bf16,
        use_cpu=config.device == "cpu",
        eval_accumulation_steps=config.eval_accumulation_steps,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        prediction_loss_only=False,
        report_to="none",
        label_names=["labels"],
        seed=config.seed,
        data_seed=config.seed,
        save_safetensors=False,
    )


def metrics_from_logits(eval_prediction):
    """Small dev metrics used for best-checkpoint selection; full reports use saved logits."""
    import numpy as np

    raw_predictions = eval_prediction.predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]
    logits = np.asarray(raw_predictions, dtype=np.float64)
    labels = np.asarray(eval_prediction.label_ids, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    accuracy = (logits.argmax(axis=1) == labels).mean()
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[labels]
    brier = ((probs - one_hot) ** 2).sum(axis=1).mean()
    return {"nll": float(nll), "accuracy": float(accuracy), "brier": float(brier)}


def create_trainer(model, tokenizer, train_examples, development_examples, config, output_dir):
    updates_per_epoch = (len(train_examples) + config.batch_size * config.accumulation - 1) // (config.batch_size * config.accumulation)
    total_steps = max(1, updates_per_epoch * config.epochs)
    cadence = min(config.save_every, total_steps)
    return DecisionTrainer(
        model=model,
        args=make_training_arguments(config, output_dir, cadence),
        train_dataset=EncodedRows(train_examples),
        eval_dataset=EncodedRows(development_examples),
        data_collator=make_collator(tokenizer, config),
        processing_class=tokenizer,
        compute_metrics=metrics_from_logits,
        callbacks=[FinalBudgetCheckpointCallback(Path(output_dir) / "final_budget_checkpoint")],
        aurc_lambda=config.aurc_lambda,
        rank_by_type=config.rank_by_type,
        final_budget_dir=Path(output_dir) / "final_budget_checkpoint",
    )
