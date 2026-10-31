"""Document Tool: turns an uploaded supplier PDF into clean, LLM-ready text.

Primary extractor is PyMuPDF (fast, preserves reading order); pypdf is used as
a fallback. The tool never raises on a bad file - it returns an
ExtractionResult with `ok=False` and a human-readable error so the
orchestrator can skip that supplier and record a warning.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

from . import config


@dataclass
class ExtractionResult:
    ok: bool
    text: str = ""
    pages: int = 0
    chars: int = 0
    truncated: bool = False
    sha256: str = ""
    extractor: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t ]+", " ", text)  # collapse spaces/tabs
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers: lines that repeat on most pages (digits ignored)."""
    if len(pages) < 2:
        return [re.sub(r"(?im)^\s*page \d+( of \d+)?\s*$", "", p) for p in pages]
    key = lambda ln: re.sub(r"\d+", "#", ln.strip().lower())
    counts: dict[str, int] = {}
    for p in pages:
        for k in {key(ln) for ln in p.splitlines() if ln.strip()}:
            counts[k] = counts.get(k, 0) + 1
    repeated = {k for k, n in counts.items() if n >= max(2, int(0.6 * len(pages))) and len(k) < 160}
    return ["\n".join(ln for ln in p.splitlines() if key(ln) not in repeated) for p in pages]


def _extract_pymupdf(data: bytes) -> tuple[str, int]:
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF releases
        import fitz as pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = _strip_repeated_lines([page.get_text("text") for page in doc])
        return "\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(pages)), len(pages)


def _extract_pypdf(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = _strip_repeated_lines([p.extract_text() or "" for p in reader.pages])
    return "\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(pages)), len(pages)


def extract_pdf_text(data: bytes, filename: str = "document.pdf") -> ExtractionResult:
    sha = hashlib.sha256(data).hexdigest()
    if not data:
        return ExtractionResult(ok=False, sha256=sha, error=f"{filename}: the file is empty.")
    if not data.lstrip().startswith(b"%PDF"):
        return ExtractionResult(ok=False, sha256=sha, error=f"{filename}: not a valid PDF file.")

    errors = []
    for name, fn in (("PyMuPDF", _extract_pymupdf), ("pypdf", _extract_pypdf)):
        try:
            raw, pages = fn(data)
        except Exception as exc:  # corrupt/encrypted PDF
            errors.append(f"{name}: {exc}")
            continue
        text = _clean(raw)
        body_chars = len(re.sub(r"\[Page \d+\]|\s", "", text))
        if body_chars < config.MIN_DOCUMENT_CHARS:
            errors.append(f"{name}: only {body_chars} readable characters")
            continue
        result = ExtractionResult(
            ok=True, text=text, pages=pages, chars=len(text), sha256=sha, extractor=name
        )
        if len(text) > config.MAX_DOCUMENT_CHARS:
            result.text = text[: config.MAX_DOCUMENT_CHARS]
            result.truncated = True
            result.warnings.append(
                f"{filename}: text truncated to {config.MAX_DOCUMENT_CHARS:,} characters for evaluation."
            )
        return result

    return ExtractionResult(
        ok=False,
        sha256=sha,
        error=(
            f"{filename}: no readable text could be extracted (it may be scanned, image-only, "
            f"encrypted or corrupt). Details: {'; '.join(errors)}"
        ),
    )
