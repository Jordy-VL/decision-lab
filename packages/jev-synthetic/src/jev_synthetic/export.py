from .models import Record
from .storage import write_rows


def trainer(record):
    r = Record.model_validate(record)
    q, s = r.question, r.source
    row = dict(id=r.id, group_id=s.group_id, source=s.dataset, split=s.split, state=s.text,
               question=q.instructions, options=q.options, target_index=q.target_index,
               type={"choice": "categorical", "noul": "boolean", "score": "ordinal"}[q.type])
    if q.type == "score":
        row["values"] = q.numeric_values
    return row


def kev(record):
    r = Record.model_validate(record)
    q = r.question
    item = {"type": q.type, "instructions": q.instructions, "src": "jev-synthetic"}
    if q.type == "choice":
        item.update(criteria={f"o{i}": option for i, option in enumerate(q.options)}, label=f"o{q.target_index}")
    elif q.type == "noul":
        item.update(label=bool(q.target_index))
    else:
        item.update(criteria=[f"{v:g}: {o}. {rubric}" for v, o, rubric in zip(q.numeric_values, q.options, q.rubric)], label=q.target_index)
    result = {"state": r.source.text, "questions": {"q": item}, "_meta": {
        "id": r.id, "group_id": r.source.group_id, "source": r.source.dataset, "split": r.source.split, "variant": "clean"}}
    validate_kev(result)
    return result


def validate_kev(record):
    if not isinstance(record["state"], str) or not record["state"] or not record["questions"]:
        raise ValueError("invalid Kev state/questions")
    for q in record["questions"].values():
        y = q["label"]
        if q["type"] == "noul":
            valid = type(y) is bool
        elif q["type"] == "choice":
            valid = isinstance(y, str) and y in q["criteria"]
        elif q["type"] == "score":
            valid = type(y) is int and 0 <= y < len(q["criteria"])
        else:
            valid = False
        if not valid:
            raise ValueError("invalid Kev label")


def exports(directory, rows):
    write_rows(directory / "trainer.jsonl", [trainer(r) for r in rows])
    write_rows(directory / "kev.jsonl", [kev(r) for r in rows])
