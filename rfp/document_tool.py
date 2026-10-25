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


def _extract_pymupdf(data: bytes) -> tuple[str, int]:
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF releases
        import fitz as pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [f"[Page {i + 1}]\n{page.get_text('text')}" for i, page in enumerate(doc)]
        return "\n".join(pages), len(pages)


def _extract_pypdf(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [f"[Page {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
    return "\n".join(pages), len(pages)


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
