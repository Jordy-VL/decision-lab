import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml


# Only names change at the YAML boundary; Config remains the single field/type/default definition.
SECTIONS = {
    "data": {"path": "data", "revision": "data_revision", "split": "data_split",
             "validation_fraction": "validation_fraction", "test_fraction": "test_fraction"},
    "model": {"name": "model", "revision": "revision", "max_length": "max_length",
              "head_hidden": "head_hidden", "max_options": "max_options"},
    "training": {key: key for key in (
        "output", "checkpoint", "resume", "save_every", "split", "seed", "batch_size", "accumulation", "epochs",
        "learning_rate", "weight_decay", "aurc_lambda", "rank_by_type", "gradient_checkpointing", "device")},
}


def flatten_yaml(values):
    """Accept either legacy flat YAML or sectioned YAML, never an ambiguous mixture."""
    nested = "training" in values or any(isinstance(values.get(key), dict) for key in ("data", "model"))
    if not nested:
        return values
    result = {}
    for section, settings in values.items():
        if section not in SECTIONS or not isinstance(settings, dict):
            raise ValueError("nested config accepts only data, model and training mappings; do not mix flat fields")
        for key, value in settings.items():
            if key not in SECTIONS[section]:
                raise ValueError(f"unknown or misplaced config field: {section}.{key}")
            result[SECTIONS[section][key]] = value
    return result


@dataclass
class Config:
    model: str = "answerdotai/ModernBERT-large"
    revision: str = "main"
    data: str = "examples/illustrative.jsonl"
    data_revision: str = "local"
    data_split: str = ""
    output: str = "runs/warmup"
    checkpoint: str = ""
    resume: str = ""
    save_every: int = 250
    split: str = "test"
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 17
    max_length: int = 2048
    head_hidden: int = 128
    max_options: int = 128
    batch_size: int = 4
    accumulation: int = 1
    epochs: int = 1
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    aurc_lambda: float = 0.0
    rank_by_type: bool = False
    gradient_checkpointing: bool = True
    device: str = "auto"

    def validate(self):
        if self.save_every < 1:
            raise ValueError("save_every must be positive")
        if self.resume and self.checkpoint:
            raise ValueError("resume and weights-only checkpoint are mutually exclusive")
        for key in ("max_length", "head_hidden", "max_options", "batch_size", "accumulation", "epochs"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.aurc_lambda <= 1:
            raise ValueError("aurc_lambda must be in [0,1]")
        if min(self.validation_fraction, self.test_fraction) < 0 or self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("split fractions must be nonnegative and sum to <1")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.max_options < 2:
            raise ValueError("max_options must be at least two")
        return self


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "predict", "check"])
    parser.add_argument("--config")
    for field in fields(Config):
        kwargs = {"default": argparse.SUPPRESS}
        if field.type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = field.type
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    args = vars(parser.parse_args(argv))
    command, path = args.pop("command"), args.pop("config")
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {} if path else {}
    if not isinstance(values, dict):
        raise ValueError("config must be a mapping")
    values = flatten_yaml(values)
    values.update(args)
    expected = {f.name: f.type for f in fields(Config)}
    for key, value in values.items():
        if key not in expected or type(value) is not expected[key]:
            # YAML whole numbers are valid for floating point fields.
            if key in expected and expected[key] is float and type(value) is int:
                values[key] = float(value)
            else:
                raise ValueError(f"unknown field or incorrect type: {key}")
    return command, Config(**values).validate()


def save_config(config, directory):
    Path(directory, "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
