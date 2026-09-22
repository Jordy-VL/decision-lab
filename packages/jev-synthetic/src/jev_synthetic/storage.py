import json
import os
from contextlib import contextmanager
from pathlib import Path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_rows(path, rows):
    atomic(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink()


class Journal:
    """Atomic rewrites of bounded JSONL: no torn append or silent tail loss."""
    def __init__(self, path):
        self.path = Path(path)
        self.rows = read_rows(path) if self.path.exists() else []

    def add(self, row):
        self.rows.append(row)
        write_rows(self.path, self.rows)
