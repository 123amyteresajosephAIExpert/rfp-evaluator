import json
from pathlib import Path

import pytest

from rfp import db

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_data"
META = json.loads((SAMPLES / "sample_metadata.json").read_text())


@pytest.fixture()
def tmp_db(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def proposal_inputs(folder="hr_platform", extra_edge=()):
    rows = [{**META[p.name], "filename": p.name, "file_bytes": p.read_bytes()}
            for p in sorted((SAMPLES / "proposals" / folder).glob("*.pdf"))]
    for f in extra_edge:
        rows.append({"supplier_name": Path(f).stem, "submission_date": "2026-10-01", "experience_rating": 3,
                     "filename": f, "file_bytes": (SAMPLES / "edge_cases" / f).read_bytes()})
    return rows
