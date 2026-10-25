"""SQLite persistence layer.

Tables
------
evaluation_criteria  - configurable criteria and weights (brief section 6)
rfp_runs             - one row per evaluation batch (brief section 6)
supplier_results     - one row per supplier per run (brief section 6)
criterion_scores     - one row per supplier x criterion (extra: full audit trail)
llm_cache            - cached validated LLM scorecards keyed by document+criteria
                       hash (extra: reproducibility and cost control)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    description   TEXT    NOT NULL,
    weight        REAL    NOT NULL CHECK (weight >= 0 AND weight <= 100),
    max_score     INTEGER NOT NULL DEFAULT 10 CHECK (max_score > 0),
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id        TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL,          -- RUNNING / COMPLETED / FAILED
    rfp_title         TEXT,
    evaluation_mode   TEXT,                   -- LLM provider/model or OFFLINE_DEMO
    criteria_snapshot TEXT,                   -- JSON copy of criteria used
    warnings_json     TEXT,
    run_json          TEXT                    -- complete exported result
);

CREATE TABLE IF NOT EXISTS supplier_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id         TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name      TEXT NOT NULL,
    submission_date    TEXT NOT NULL,
    experience_rating  REAL NOT NULL,
    absolute_score     REAL,
    ppi                REAL,
    final_rank         INTEGER,
    status             TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING / RANKED / NOT_EVALUATED
    result_json        TEXT NOT NULL,
    UNIQUE (rfp_run_id, supplier_name)
);

CREATE TABLE IF NOT EXISTS criterion_scores (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id           TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name        TEXT NOT NULL,
    criterion_id         INTEGER NOT NULL,
    criterion_name       TEXT NOT NULL,
    weight               REAL NOT NULL,
    max_score            INTEGER NOT NULL,
    score                REAL NOT NULL,
    benchmark_score      REAL,
    gap                  REAL,
    relative_pct         REAL,
    weighted_points      REAL,
    justification        TEXT,
    evidence             TEXT,
    evidence_verified    INTEGER
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    model        TEXT NOT NULL,
    response     TEXT NOT NULL
);
"""

DEFAULT_CRITERIA = [
    (
        "Technical Capability",
        "Architecture, integrations, scalability and overall technical fit to the requirement.",
        30,
    ),
    (
        "Implementation Plan",
        "Timeline, milestones, staffing, governance and risk/mitigation plan.",
        20,
    ),
    (
        "Commercial Value",
        "Pricing clarity, total cost of ownership, stated assumptions and exclusions.",
        20,
    ),
    (
        "Security & Compliance",
        "Security controls, certifications (e.g. ISO 27001, SOC 2), data privacy and auditability.",
        20,
    ),
    (
        "Support & Experience",
        "Support model and SLAs, similar past projects and client references.",
        10,
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None, seed: bool = True) -> None:
    """Create tables (idempotent) and seed default criteria if empty."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(supplier_results)")}
        if "status" not in cols:  # upgrade databases created by an earlier version
            conn.execute("ALTER TABLE supplier_results ADD COLUMN status TEXT NOT NULL DEFAULT 'PENDING'")
        count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if seed and count == 0:
            now = utc_now()
            conn.executemany(
                "INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active, updated_at)"
                " VALUES (?, ?, ?, 10, 1, ?)",
                [(n, d, w, now) for n, d, w in DEFAULT_CRITERIA],
            )


# --- Criteria -------------------------------------------------------------------


def get_criteria(db_path=None, active_only: bool = True) -> list[dict]:
    sql = "SELECT criterion_id, name, description, weight, max_score, is_active FROM evaluation_criteria"
    if active_only:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY criterion_id"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def validate_criteria_set(criteria: list[dict]) -> list[str]:
    """Return a list of problems with a proposed criteria configuration."""
    problems: list[str] = []
    active = [c for c in criteria if int(c.get("is_active", 0)) == 1]
    if not active:
        problems.append("At least one criterion must be active.")
    total = round(sum(float(c["weight"]) for c in active), 6)
    if abs(total - 100) > 1e-6:
        problems.append(f"Active criteria weights must total 100% (currently {total:g}%).")
    names = [str(c["name"]).strip().lower() for c in criteria]
    if len(names) != len(set(names)):
        problems.append("Criterion names must be unique.")
    for c in criteria:
        if not str(c.get("name", "")).strip():
            problems.append("Every criterion needs a name.")
        if float(c["weight"]) < 0:
            problems.append(f"'{c['name']}' has a negative weight.")
        if int(c["max_score"]) <= 0:
            problems.append(f"'{c['name']}' needs a maximum score above zero.")
    return problems


def save_criteria(criteria: list[dict], db_path=None) -> None:
    """Upsert the full criteria table after validation (raises ValueError)."""
    problems = validate_criteria_set(criteria)
    if problems:
        raise ValueError(" ".join(problems))
    now = utc_now()
    with connect(db_path) as conn:
        for c in criteria:
            if c.get("criterion_id"):
                conn.execute(
                    "UPDATE evaluation_criteria SET name=?, description=?, weight=?, max_score=?,"
                    " is_active=?, updated_at=? WHERE criterion_id=?",
                    (
                        c["name"].strip(),
                        c["description"].strip(),
                        float(c["weight"]),
                        int(c["max_score"]),
                        int(c["is_active"]),
                        now,
                        int(c["criterion_id"]),
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        c["name"].strip(),
                        c["description"].strip(),
                        float(c["weight"]),
                        int(c["max_score"]),
                        int(c["is_active"]),
                        now,
                    ),
                )


# --- Runs and results -------------------------------------------------------------


def create_run(rfp_run_id: str, rfp_title: str, mode: str, criteria: list[dict],
               suppliers: list[dict] | None = None, db_path=None) -> None:
    """Create the batch: one rfp_runs row plus a PENDING supplier entry per uploaded document."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO rfp_runs (rfp_run_id, created_at, status, rfp_title, evaluation_mode, criteria_snapshot)"
            " VALUES (?, ?, 'RUNNING', ?, ?, ?)",
            (rfp_run_id, utc_now(), rfp_title, mode, json.dumps(criteria)),
        )
        for s in suppliers or []:
            conn.execute(
                "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating,"
                " status, result_json) VALUES (?, ?, ?, ?, 'PENDING', ?)",
                (rfp_run_id, s["supplier_name"], str(s["submission_date"]), float(s["experience_rating"]),
                 json.dumps({"source_file": s.get("filename")})),
            )


def fail_run(rfp_run_id: str, warnings: list[str], db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE rfp_runs SET status='FAILED', warnings_json=? WHERE rfp_run_id=?",
            (json.dumps(warnings), rfp_run_id),
        )


def save_run_results(run: dict, db_path=None) -> None:
    """Write the complete result of a run in a single transaction."""
    run_id = run["rfp_run_id"]
    with connect(db_path) as conn:
        conn.execute("DELETE FROM criterion_scores WHERE rfp_run_id=?", (run_id,))
        for n in run.get("not_evaluated", []):
            conn.execute(
                "UPDATE supplier_results SET status='NOT_EVALUATED', result_json=? WHERE rfp_run_id=? AND supplier_name=?",
                (json.dumps(n), run_id, n["supplier_name"]),
            )
        for s in run["suppliers"]:
            conn.execute(
                "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating,"
                " absolute_score, ppi, final_rank, status, result_json) VALUES (?, ?, ?, ?, ?, ?, ?, 'RANKED', ?)"
                " ON CONFLICT(rfp_run_id, supplier_name) DO UPDATE SET submission_date=excluded.submission_date,"
                " experience_rating=excluded.experience_rating, absolute_score=excluded.absolute_score,"
                " ppi=excluded.ppi, final_rank=excluded.final_rank, status='RANKED', result_json=excluded.result_json",
                (
                    run_id,
                    s["supplier_name"],
                    s["submission_date"],
                    s["experience_rating"],
                    s["absolute_score"],
                    s["ppi"],
                    s["final_rank"],
                    json.dumps(s),
                ),
            )
            for c in s["criteria"]:
                conn.execute(
                    "INSERT INTO criterion_scores (rfp_run_id, supplier_name, criterion_id, criterion_name, weight,"
                    " max_score, score, benchmark_score, gap, relative_pct, weighted_points, justification,"
                    " evidence, evidence_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        s["supplier_name"],
                        c["criterion_id"],
                        c["name"],
                        c["weight"],
                        c["max_score"],
                        c["score"],
                        c["benchmark_score"],
                        c["gap"],
                        c["relative_pct"],
                        c["weighted_points"],
                        c["justification"],
                        c["evidence"],
                        int(bool(c["evidence_verified"])),
                    ),
                )
        conn.execute(
            "UPDATE rfp_runs SET status='COMPLETED', warnings_json=?, run_json=? WHERE rfp_run_id=?",
            (json.dumps(run["warnings"]), json.dumps(run), run_id),
        )


def list_runs(db_path=None, limit: int = 50) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT r.rfp_run_id, r.created_at, r.status, r.rfp_title, r.evaluation_mode,"
            " (SELECT COUNT(*) FROM supplier_results s WHERE s.rfp_run_id = r.rfp_run_id AND s.status = 'RANKED') AS suppliers"
            " FROM rfp_runs r ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def load_run(rfp_run_id: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT run_json FROM rfp_runs WHERE rfp_run_id=?", (rfp_run_id,)).fetchone()
        if row and row["run_json"]:
            return json.loads(row["run_json"])
        return None


# --- LLM cache ------------------------------------------------------------------------


def cache_get(cache_key: str, db_path=None) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT response FROM llm_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return row["response"] if row else None


def cache_put(cache_key: str, model: str, response: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, created_at, model, response) VALUES (?, ?, ?, ?)",
            (cache_key, utc_now(), model, response),
        )
