"""Batch PP-OCRv6 over RVL-CDIP-N_MultiPage PDFs on Modal.

The default Modal run is a bounded pilot: one test PDF per class. Pass
--limit-per-class 0 to continue over the complete test split after reviewing
the pilot. OCR artifacts are written to the existing decision-lab-profile
Volume and never enter training.

Examples (from the decision-lab repository root):
    uv run modal run scripts/modal_rvl_cdip_n_mp_ocr.py
    uv run modal run scripts/modal_rvl_cdip_n_mp_ocr.py --limit-per-class 0
    uv run modal run scripts/modal_rvl_cdip_n_mp_ocr.py --limit-per-class 0 --max-pages 5
"""
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
import hashlib
import modal

ROOT = Path(__file__).resolve().parents[1]
OCR_SOURCE = Path("/home/jvl/tinker/src/ppocrv6-mps/ppocrv6_ocr_mps.py")
app = modal.App("decision-lab-rvl-cdip-n-mp-ocr")
volume = modal.Volume.from_name("decision-lab-profile", create_if_missing=False)

DATASET_REPO = "jordyvl/rvl_cdip_n_mp"
DATASET_REVISION = "a0668401c9f46b157c4f37694fdc96e8506453fa"
OCR_REVISIONS = {
    "PaddlePaddle/PP-OCRv6_medium_det": "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
    "PaddlePaddle/PP-OCRv6_medium_rec": "e5a92bcbc5cc1b494628e458d267778f0704fd7c",
}
CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement", "scientific report",
    "scientific publication", "specification", "file folder", "news article", "budget",
    "invoice", "presentation", "questionnaire", "resume", "memo",
]

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0")
         .uv_pip_install(
             "torch==2.8.0", "onnx>=1.16", "onnx2torch>=1.5.15", "numpy<2",
             "opencv-python-headless>=4.8", "pyclipper>=1.3", "shapely>=2.0",
             "pymupdf>=1.24", "pyyaml>=6.0", "huggingface-hub>=0.25", "uv",
         )
         .env({"HF_HOME": "/artifacts/hf-cache", "PYTHONUNBUFFERED": "1"})
         .add_local_file(OCR_SOURCE, "/workspace/ppocrv6_ocr_mps.py"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_pdfs(archive_path: Path, pdf_root: Path):
    """Extract only PDF members, rejecting archive paths outside the target."""
    pdf_root.mkdir(parents=True, exist_ok=True)
    root = pdf_root.resolve()
    found = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [m for m in archive.getmembers()
                   if m.isfile() and PurePosixPath(m.name).suffix.lower() == ".pdf"]
        for member in members:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe PDF archive path: {member.name}")
            destination = (pdf_root / Path(*rel.parts)).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"PDF archive path escapes output directory: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            found.append(destination)
    return sorted(found)


def _download_archive(destination: Path) -> Path:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    cached = Path(hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename="data.tar.gz",
        revision=DATASET_REVISION,
    ))
    shutil.copyfile(cached, destination)
    return destination


def _prepare_ocr_models(ocr, cache_dir: Path, device):
    """Pre-fetch immutable model revisions, then reuse the existing converter."""
    from huggingface_hub import hf_hub_download

    for repo, tag in ((ocr.DET_REPO, "det"), (ocr.REC_REPO, "rec")):
        model_dir = cache_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in ocr.PADDLE_FILES:
            cached = Path(hf_hub_download(
                repo_id=repo, filename=filename, revision=OCR_REVISIONS[repo],
            ))
            shutil.copyfile(cached, model_dir / filename)
    det_model = ocr.build_model(ocr.DET_REPO, "det", cache_dir, device)
    rec_model = ocr.build_model(ocr.REC_REPO, "rec", cache_dir, device)
    charset = ocr.load_charset(cache_dir)
    return det_model, rec_model, charset


def _ocr_pdf(ocr, pdf_path: Path, rel_path: str, det_model, rec_model, charset,
             device, dpi: int, limit_side_len: int, rec_batch_size: int,
             drop_score: float, max_pages: int | None):
    import numpy as np
    import torch

    page_rows, text_pages = [], []
    for page_index, img in ocr.load_pages(pdf_path, dpi, max_pages):
        height, width = img.shape[:2]
        tensor, _ = ocr.preprocess_det(img, limit_side_len, "min")
        with torch.inference_mode():
            prediction = det_model(tensor.to(device))
            torch.cuda.synchronize(device)
        probability = prediction[0, 0].float().cpu().numpy()
        boxes, det_scores = ocr.boxes_from_bitmap(
            probability, probability > ocr.DB_THRESH, width, height,
        )
        boxes, det_scores = ocr.sort_boxes(boxes, det_scores)
        recognized = ocr.recognize(
            rec_model, [ocr.crop_quad(img, box) for box in boxes], charset,
            device, rec_batch_size,
        ) if boxes else []
        keep = [i for i, (_, score) in enumerate(recognized) if score >= drop_score]
        lines = [
            {"text": recognized[i][0], "det_score": round(det_scores[i], 4),
             "rec_score": round(recognized[i][1], 4), "box": boxes[i].tolist()}
            for i in keep
        ]
        page_text = "\n".join(line["text"] for line in lines)
        text_pages.append(f"--- page {page_index + 1} ---\n{page_text}")
        page_rows.append({"page": page_index + 1, "width": width,
                          "height": height, "lines": lines})

    return {
        "id": rel_path,
        "source": rel_path,
        "label": pdf_path.parent.name.lower(),
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "split": "test",
        "ocr": {
            "engine": "PP-OCRv6 medium det+rec",
            "det_revision": OCR_REVISIONS[ocr.DET_REPO],
            "rec_revision": OCR_REVISIONS[ocr.REC_REPO],
            "dpi": dpi, "limit_side_len": limit_side_len,
            "recognition_drop_score": drop_score,
        },
        "pages": page_rows,
        "text": "\n\n".join(text_pages),
    }


@app.function(image=image, gpu="A10G", cpu=4, memory=32768,
              volumes={"/artifacts": volume}, timeout=7200, retries=0,
              max_containers=1, scaledown_window=2)
def batch_ocr(run_id: str, limit_per_class: int = 1, max_pages: int = 0,
              dpi: int = 200, force: bool = False):
    import importlib.util
    import torch

    started = time.perf_counter()
    base = Path("/artifacts/datasets/rvl_cdip_n_mp")
    archive_path = _download_archive(base / "source/data.tar.gz")
    archive_sha256 = _sha256(archive_path)
    pdf_root = base / "pdfs"
    marker = pdf_root / ".extracted.json"
    if not marker.exists() or json.loads(marker.read_text()).get("archive_sha256") != archive_sha256:
        if pdf_root.exists():
            shutil.rmtree(pdf_root)
        pdfs = _safe_extract_pdfs(archive_path, pdf_root)
        if len(pdfs) != 998:
            raise ValueError(f"expected 998 PDFs (including 7 out-of-scope language files), found {len(pdfs)}")
        marker.write_text(json.dumps({"archive_sha256": archive_sha256,
                                      "archive_pdf_count": len(pdfs)}, indent=2))
        volume.commit()
    else:
        pdfs = sorted(pdf_root.rglob("*.pdf"))

    by_class = {name: [] for name in CLASSES}
    excluded_by_directory = {}
    for path in pdfs:
        label = path.parent.name.lower()
        if label not in by_class:
            excluded_by_directory[label] = excluded_by_directory.get(label, 0) + 1
        else:
            by_class[label].append(path)
    counts = {name: len(paths) for name, paths in by_class.items()}
    if len(pdfs) != 998 or sum(counts.values()) != 991 or excluded_by_directory != {"language": 7}:
        raise ValueError(f"unexpected archive contents: labeled={sum(counts.values())}, "
                         f"excluded={excluded_by_directory}, class_counts={counts}")
    populated_classes = [name for name in CLASSES if by_class[name]]

    if limit_per_class > 0:
        selected = [path for label in populated_classes
                    for path in sorted(by_class[label])[:limit_per_class]]
    else:
        selected = sorted(path for paths in by_class.values() for path in paths)

    cache_dir = Path("/artifacts/models/ppocrv6")
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("ppocrv6_ocr_mps", "/workspace/ppocrv6_ocr_mps.py")
    ocr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr)
    device = torch.device("cuda")
    det_model, rec_model, charset = _prepare_ocr_models(ocr, cache_dir, device)

    out = base / "ocr/ppocrv6-medium"
    out.mkdir(parents=True, exist_ok=True)
    run_history = out / "runs"
    run_history.mkdir(parents=True, exist_ok=True)
    run_record = run_history / f"{run_id}.json"
    run_meta = {
        "run_id": run_id, "status": "running", "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION, "dataset_archive_sha256": archive_sha256,
        "split": "test", "class_counts": counts,
        "populated_class_count": len(populated_classes),
        "declared_but_empty_classes": [name for name, n in counts.items() if n == 0],
        "excluded_out_of_scope_directories": excluded_by_directory,
        "selected_documents": len(selected),
        "limit_per_class": limit_per_class, "max_pages": max_pages or None,
        "dpi": dpi, "device": torch.cuda.get_device_name(),
        "ocr_engine": "PP-OCRv6 medium det+rec",
        "det_revision": OCR_REVISIONS[ocr.DET_REPO],
        "rec_revision": OCR_REVISIONS[ocr.REC_REPO],
        "completed_documents": 0, "failed_documents": [],
    }

    def save_run_meta():
        payload = json.dumps(run_meta, indent=2)
        (out / "run.json").write_text(payload)
        run_record.write_text(payload)

    for i, pdf_path in enumerate(selected, 1):
        # Modal's mounted-volume path may resolve to /artifacts or its backing
        # /__modal/volumes path depending on the container; labels are the
        # immediate parent directory in this archive, so construct a stable ID.
        rel_path = f"{pdf_path.parent.name}/{pdf_path.name}"
        key = hashlib.sha256(rel_path.encode()).hexdigest()[:24]
        result_path = out / f"{key}.json"
        if result_path.exists() and not force:
            try:
                existing = json.loads(result_path.read_text())
                if existing.get("dataset_revision") == DATASET_REVISION:
                    run_meta["completed_documents"] += 1
                    continue
            except (OSError, ValueError):
                pass
        error_path = out / f"{key}.error.json"
        error_path.unlink(missing_ok=True)
        try:
            result = _ocr_pdf(
                ocr, pdf_path, rel_path, det_model, rec_model, charset, device,
                dpi=dpi, limit_side_len=736, rec_batch_size=8, drop_score=0.5,
                max_pages=max_pages or None,
            )
            result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            error_path.unlink(missing_ok=True)
            run_meta["completed_documents"] += 1
            page_count = len(result["pages"])
            line_count = sum(len(page["lines"]) for page in result["pages"])
            print(json.dumps({"document": i, "total": len(selected), "id": rel_path,
                              "pages": page_count, "lines": line_count}), flush=True)
        except Exception as exc:
            error_path.write_text(json.dumps({"id": rel_path, "error": repr(exc)}, indent=2))
            run_meta["failed_documents"].append({"id": rel_path, "error": repr(exc)})
            print(json.dumps({"document": i, "total": len(selected), "id": rel_path,
                              "error": repr(exc)}), flush=True)
        if i % 8 == 0:
            run_meta["elapsed_seconds"] = time.perf_counter() - started
            save_run_meta()
            volume.commit()

    run_meta["elapsed_seconds"] = time.perf_counter() - started
    run_meta["status"] = "complete" if not run_meta["failed_documents"] else "partial"
    save_run_meta()
    manifest_rows = []
    for path in sorted(out.glob("*.json")):
        if path.name == "run.json" or path.name.endswith(".error.json"):
            continue
        item = json.loads(path.read_text())
        manifest_rows.append({"id": item["id"], "label": item["label"],
                              "pages": len(item["pages"]), "ocr_file": path.name,
                              "text_file": path.name.removesuffix(".json") + ".txt"})
        path.with_suffix(".txt").write_text(item["text"], encoding="utf-8")
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    volume.commit()
    print(json.dumps(run_meta, indent=2), flush=True)
    return run_meta


@app.function(image=image, cpu=1, memory=2048,
              volumes={"/artifacts": volume}, timeout=300,
              retries=0, max_containers=1, scaledown_window=2)
def inspect_pdfs():
    root = Path("/artifacts/datasets/rvl_cdip_n_mp/pdfs")
    pdfs = sorted(root.rglob("*.pdf"))
    counts = {}
    for path in pdfs:
        counts[path.parent.name.lower()] = counts.get(path.parent.name.lower(), 0) + 1
    result = {"pdf_count": len(pdfs), "class_directory_counts": counts}
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(limit_per_class: int = 1, max_pages: int = 0, dpi: int = 200,
         force: bool = False):
    run_id = "rvlcdipnmp-ocr-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(json.dumps(batch_ocr.remote(run_id, limit_per_class, max_pages, dpi, force), indent=2))
