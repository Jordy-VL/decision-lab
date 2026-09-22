"""Validate exports against an existing Kev checkout without loading training code."""
import argparse
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kev-root", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    a = p.parse_args()
    split = module("kev_split", a.kev_root / "skills/kev-finetune/scripts/split_data.py")
    api = module("kev_api", a.kev_root / "kev/api.py")
    count = 0
    for line in a.data.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        errors = split.check_record(r)
        if errors:
            raise ValueError(errors)
        request = api.SystemOneRequest.model_validate(r)
        internal, meta = api.to_record(request)
        for qid, q in r["questions"].items():
            keys = api.question_keys(q["type"], q.get("criteria"))
            index = keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])
            assert 0 <= index < len(internal["questions"][0]["options"])
        count += 1
    print(json.dumps({"upstream_validated_records": count}))


if __name__ == "__main__":
    main()
