"""SQLite persistence: the single source of truth for templates, criteria, prompts,
runs, results, proposal text, evaluation DNA and analyst overrides.

Nothing is passed between modules only in memory: every run's inputs and outputs are
written here, so any past run can be reopened, audited or re-scored later.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import config, prompts

SCHEMA = """
CREATE TABLE IF NOT EXISTS rfp_templates (
    template_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL UNIQUE,
    buyer             TEXT,
    industry          TEXT,
    summary           TEXT,
    requirements_json TEXT,
    source_filename   TEXT,
    source_sha256     TEXT,
    template_text     TEXT,
    parse_mode        TEXT,                    -- SEEDED / LLM / RULES / MANUAL
    created_at        TEXT NOT NULL,
    is_archived       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id        INTEGER NOT NULL REFERENCES rfp_templates(template_id),
    name               TEXT NOT NULL,
    description        TEXT NOT NULL,
    industry_benchmark TEXT NOT NULL DEFAULT '',
    weight             REAL NOT NULL CHECK (weight >= 0 AND weight <= 100),
    max_score          INTEGER NOT NULL DEFAULT 10 CHECK (max_score > 0),
    is_active          INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    updated_at         TEXT NOT NULL,
    UNIQUE (template_id, name)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name TEXT NOT NULL,
    version     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    note        TEXT,
    created_at  TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 0,
    UNIQUE (prompt_name, version)
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id        TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT,
    status            TEXT NOT NULL,           -- RUNNING / COMPLETED / FAILED
    template_id       INTEGER,
    template_name     TEXT,
    rfp_title         TEXT,
    evaluation_mode   TEXT,
    criteria_snapshot TEXT,
    prompt_snapshot   TEXT,
    warnings_json     TEXT,
    run_json          TEXT,
    revision          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS supplier_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id         TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name      TEXT NOT NULL,
    submission_date    TEXT NOT NULL,
    experience_rating  REAL NOT NULL,
    absolute_score     REAL,
    ppi                REAL,
    industry_index     REAL,
    final_rank         INTEGER,
    status             TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING / RANKED / NOT_EVALUATED
    result_json        TEXT NOT NULL,
    UNIQUE (rfp_run_id, supplier_name)
);

CREATE TABLE IF NOT EXISTS criterion_scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id        TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name     TEXT NOT NULL,
    criterion_id      INTEGER NOT NULL,
    criterion_name    TEXT NOT NULL,
    weight            REAL NOT NULL,
    max_score         INTEGER NOT NULL,
    score             REAL NOT NULL,
    industry_score    REAL,
    benchmark_score   REAL,
    gap               REAL,
    relative_pct      REAL,
    weighted_points   REAL,
    justification     TEXT,
    evidence          TEXT,
    evidence_verified INTEGER,
    overridden        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_documents (
    rfp_run_id    TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name TEXT NOT NULL,
    filename      TEXT,
    document_sha  TEXT,
    pages         INTEGER,
    chars         INTEGER,
    text          TEXT,
    PRIMARY KEY (rfp_run_id, supplier_name)
);

CREATE TABLE IF NOT EXISTS evaluation_dna (
    fingerprint    TEXT PRIMARY KEY,
    document_sha   TEXT NOT NULL,
    criteria_hash  TEXT NOT NULL,
    prompt_hash    TEXT NOT NULL,
    model          TEXT NOT NULL,
    response       TEXT NOT NULL,
    first_run_id   TEXT,
    first_supplier TEXT,
    created_at     TEXT NOT NULL,
    last_used_at   TEXT,
    reuse_count    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS score_overrides (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id     TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id),
    supplier_name  TEXT NOT NULL,
    criterion_id   INTEGER NOT NULL,
    criterion_name TEXT NOT NULL,
    old_score      REAL NOT NULL,
    new_score      REAL NOT NULL,
    reason         TEXT NOT NULL,
    analyst        TEXT,
    created_at     TEXT NOT NULL
);
"""

# ------------------------------------------------------------------------------------
# Seed data: the sample RFP template and its five criteria from the project brief
# ------------------------------------------------------------------------------------
SAMPLE_TEMPLATE = {
    "name": "Kestrel - HR Service Delivery Platform (sample)",
    "buyer": "Kestrel Manufacturing Ltd (fictional)",
    "industry": "Enterprise HR technology / SaaS",
    "summary": ("Kestrel, a 6,500-employee manufacturer with 4 plants and a head office, wants a cloud HR service "
                "delivery platform: employee self-service portal, HR case management with SLAs, a policy "
                "knowledge base, and integration with SAP SuccessFactors and Microsoft Entra ID single sign-on."),
    "requirements": [
        "Employee portal usable by office and frontline (plant) employees",
        "HR case management with SLA tracking and escalation",
        "Integration with SAP SuccessFactors HRIS and Microsoft Entra ID SSO",
        "India data residency and DPDP Act 2023 compliance",
        "Go-live across all sites within about 5 months",
        "Three-year total cost of ownership with clear assumptions",
    ],
    "criteria": [
        ("Technical Capability", 30,
         "Architecture, integrations, scalability and overall technical fit to the requirement.",
         "Cloud-native platform with certified HRIS connectors, open REST APIs, SSO/SCIM, 99.9%+ availability "
         "and proven load testing at several times the client's headcount."),
        ("Implementation Plan", 20,
         "Timeline, milestones, staffing, governance and risk/mitigation plan.",
         "Phased plan with dated milestones and exit criteria, named team, pilot before roll-out, RAID log and "
         "hypercare; typical HR service delivery go-live is 12-20 weeks."),
        ("Commercial Value", 20,
         "Pricing clarity, total cost of ownership, stated assumptions and exclusions.",
         "Transparent 3-year TCO, fixed-price or milestone-linked implementation, price locks, and all "
         "assumptions and exclusions stated - no open-ended estimates."),
        ("Security & Compliance", 20,
         "Security controls, certifications (e.g. ISO 27001, SOC 2), data privacy and auditability.",
         "Current ISO 27001 and SOC 2 Type II, encryption at rest and in transit, local data residency, "
         "DPDP/GDPR processes, MFA, immutable audit logs and independent penetration testing."),
        ("Support & Experience", 10,
         "Support model and SLAs, similar past projects and client references.",
         "24x7 P1 support with response SLAs and service credits, a customer success manager, and several "
         "referenceable implementations of similar size and industry."),
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
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


def init_db(db_path=None, seed: bool = True) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
    if seed and not list_templates(db_path, include_archived=True):
        t = SAMPLE_TEMPLATE
        save_template(
            {"name": t["name"], "buyer": t["buyer"], "industry": t["industry"], "summary": t["summary"],
             "requirements": t["requirements"], "parse_mode": "SEEDED"},
            [{"name": n, "weight": w, "max_score": 10, "description": d, "industry_benchmark": b, "is_active": 1}
             for n, w, d, b in t["criteria"]],
            db_path,
        )


# ------------------------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------------------------


def _template_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["requirements"] = json.loads(d.pop("requirements_json") or "[]")
    return d


def list_templates(db_path=None, include_archived: bool = False) -> list[dict]:
    sql = "SELECT * FROM rfp_templates" + ("" if include_archived else " WHERE is_archived = 0") + " ORDER BY template_id"
    with connect(db_path) as conn:
        return [_template_row(r) for r in conn.execute(sql).fetchall()]


def get_template(template_id: int, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM rfp_templates WHERE template_id=?", (template_id,)).fetchone()
        return _template_row(r) if r else None


def find_template_by_sha(sha: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM rfp_templates WHERE source_sha256=? AND is_archived=0", (sha,)).fetchone()
        return _template_row(r) if r else None


def save_template(template: dict, criteria: list[dict], db_path=None) -> int:
    """Insert a new template with its criteria (validated). Returns template_id."""
    problems = validate_criteria_set(criteria)
    if problems:
        raise ValueError(" ".join(problems))
    name = template["name"].strip()
    now = utc_now()
    with connect(db_path) as conn:
        existing = {r["name"] for r in conn.execute("SELECT name FROM rfp_templates")}
        base, i = name, 2
        while name in existing:
            name = f"{base} ({i})"
            i += 1
        cur = conn.execute(
            "INSERT INTO rfp_templates (name, buyer, industry, summary, requirements_json, source_filename,"
            " source_sha256, template_text, parse_mode, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, template.get("buyer"), template.get("industry"), template.get("summary"),
             json.dumps(template.get("requirements", [])), template.get("source_filename"),
             template.get("source_sha256"), template.get("template_text"), template.get("parse_mode", "MANUAL"), now),
        )
        tid = cur.lastrowid
        for c in criteria:
            conn.execute(
                "INSERT INTO evaluation_criteria (template_id, name, description, industry_benchmark, weight,"
                " max_score, is_active, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tid, c["name"].strip(), str(c.get("description") or "").strip(),
                 str(c.get("industry_benchmark") or "").strip(), float(c["weight"]),
                 int(c.get("max_score") or config.DEFAULT_MAX_SCORE), int(c.get("is_active", 1)), now),
            )
        return tid


def update_template_info(template_id: int, fields: dict, db_path=None) -> None:
    allowed = {"name", "buyer", "industry", "summary"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if "requirements" in fields:
        sets["requirements_json"] = json.dumps(fields["requirements"])
    if not sets:
        return
    with connect(db_path) as conn:
        conn.execute(f"UPDATE rfp_templates SET {', '.join(k + '=?' for k in sets)} WHERE template_id=?",
                     (*sets.values(), template_id))


def archive_template(template_id: int, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE rfp_templates SET is_archived=1 WHERE template_id=?", (template_id,))


# ------------------------------------------------------------------------------------
# Criteria
# ------------------------------------------------------------------------------------


def get_criteria(template_id: int, db_path=None, active_only: bool = True) -> list[dict]:
    sql = ("SELECT criterion_id, template_id, name, description, industry_benchmark, weight, max_score, is_active"
           " FROM evaluation_criteria WHERE template_id=?")
    if active_only:
        sql += " AND is_active = 1"
    sql += " ORDER BY criterion_id"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, (template_id,)).fetchall()]


def validate_criteria_set(criteria: list[dict]) -> list[str]:
    problems: list[str] = []
    active = [c for c in criteria if int(c.get("is_active", 1)) == 1]
    if len(active) < config.MIN_CRITERIA:
        problems.append(f"At least {config.MIN_CRITERIA} criteria must be active.")
    if len(active) > config.MAX_CRITERIA:
        problems.append(f"Use at most {config.MAX_CRITERIA} active criteria.")
    total = round(sum(float(c.get("weight") or 0) for c in active), 6)
    if abs(total - 100) > 1e-6:
        problems.append(f"Active criteria weights must total 100% (currently {total:g}%).")
    names = [str(c.get("name", "")).strip().lower() for c in criteria]
    if any(not n for n in names):
        problems.append("Every criterion needs a name.")
    if len(names) != len(set(names)):
        problems.append("Criterion names must be unique.")
    for c in criteria:
        if float(c.get("weight") or 0) < 0:
            problems.append(f"'{c.get('name')}' has a negative weight.")
        if int(c.get("max_score") or 0) <= 0:
            problems.append(f"'{c.get('name')}' needs a maximum score above zero.")
    return problems


def save_criteria(template_id: int, criteria: list[dict], db_path=None) -> None:
    problems = validate_criteria_set(criteria)
    if problems:
        raise ValueError(" ".join(problems))
    now = utc_now()
    with connect(db_path) as conn:
        for c in criteria:
            values = (c["name"].strip(), str(c.get("description") or "").strip(),
                      str(c.get("industry_benchmark") or "").strip(), float(c["weight"]),
                      int(c["max_score"]), int(c["is_active"]), now)
            if c.get("criterion_id"):
                conn.execute("UPDATE evaluation_criteria SET name=?, description=?, industry_benchmark=?, weight=?,"
                             " max_score=?, is_active=?, updated_at=? WHERE criterion_id=? AND template_id=?",
                             (*values, int(c["criterion_id"]), template_id))
            else:
                conn.execute("INSERT INTO evaluation_criteria (name, description, industry_benchmark, weight,"
                             " max_score, is_active, updated_at, template_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (*values, template_id))


# ------------------------------------------------------------------------------------
# Prompts (versioned)
# ------------------------------------------------------------------------------------


def get_prompt(name: str, db_path=None) -> dict:
    """Active version of a prompt, or the built-in default (version 0)."""
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM prompt_versions WHERE prompt_name=? AND is_active=1", (name,)).fetchone()
    if r:
        return dict(r)
    return {"id": None, "prompt_name": name, "version": 0, "text": prompts.PROMPTS[name][1],
            "note": "Built-in default", "created_at": None, "is_active": 1}


def list_prompt_versions(name: str, db_path=None) -> list[dict]:
    with connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM prompt_versions WHERE prompt_name=? ORDER BY version DESC", (name,)).fetchall()]
    return rows


def save_prompt_version(name: str, text: str, note: str = "", db_path=None) -> int:
    problems = prompts.check_prompt(name, text)
    if problems:
        raise ValueError(" ".join(problems))
    with connect(db_path) as conn:
        v = conn.execute("SELECT COALESCE(MAX(version), 0) FROM prompt_versions WHERE prompt_name=?", (name,)).fetchone()[0] + 1
        conn.execute("UPDATE prompt_versions SET is_active=0 WHERE prompt_name=?", (name,))
        conn.execute("INSERT INTO prompt_versions (prompt_name, version, text, note, created_at, is_active)"
                     " VALUES (?, ?, ?, ?, ?, 1)", (name, v, text, note, utc_now()))
        return v


def activate_prompt_version(name: str, version: int, db_path=None) -> None:
    """version 0 = go back to the built-in default."""
    with connect(db_path) as conn:
        conn.execute("UPDATE prompt_versions SET is_active=0 WHERE prompt_name=?", (name,))
        if version:
            conn.execute("UPDATE prompt_versions SET is_active=1 WHERE prompt_name=? AND version=?", (name, version))


# ------------------------------------------------------------------------------------
# Runs and results
# ------------------------------------------------------------------------------------


def create_run(rfp_run_id: str, template: dict, mode: str, criteria: list[dict], prompt_snapshot: dict,
               suppliers: list[dict], db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO rfp_runs (rfp_run_id, created_at, updated_at, status, template_id, template_name, rfp_title,"
            " evaluation_mode, criteria_snapshot, prompt_snapshot) VALUES (?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?)",
            (rfp_run_id, utc_now(), utc_now(), template["template_id"], template["name"], template["name"], mode,
             json.dumps(criteria), json.dumps(prompt_snapshot)),
        )
        for s in suppliers:
            conn.execute(
                "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating,"
                " status, result_json) VALUES (?, ?, ?, ?, 'PENDING', ?)",
                (rfp_run_id, s["supplier_name"], str(s["submission_date"]), float(s["experience_rating"]),
                 json.dumps({"source_file": s.get("filename")})),
            )


def fail_run(rfp_run_id: str, warnings: list[str], db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE rfp_runs SET status='FAILED', warnings_json=?, updated_at=? WHERE rfp_run_id=?",
                     (json.dumps(warnings), utc_now(), rfp_run_id))


def save_run_documents(rfp_run_id: str, docs: dict, db_path=None) -> None:
    with connect(db_path) as conn:
        for name, d in docs.items():
            conn.execute(
                "INSERT OR REPLACE INTO run_documents (rfp_run_id, supplier_name, filename, document_sha, pages,"
                " chars, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rfp_run_id, name, d["filename"], d["sha256"], d["pages"], d["chars"], d["text"]),
            )


def get_run_documents(rfp_run_id: str, db_path=None) -> dict:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM run_documents WHERE rfp_run_id=?", (rfp_run_id,)).fetchall()
        return {r["supplier_name"]: dict(r) for r in rows}


def save_run_results(run: dict, db_path=None) -> None:
    """Write the complete result of a run (or a re-scored revision) in one transaction."""
    run_id = run["rfp_run_id"]
    with connect(db_path) as conn:
        conn.execute("DELETE FROM criterion_scores WHERE rfp_run_id=?", (run_id,))
        for n in run.get("not_evaluated", []):
            conn.execute("UPDATE supplier_results SET status='NOT_EVALUATED', result_json=?"
                         " WHERE rfp_run_id=? AND supplier_name=?", (json.dumps(n), run_id, n["supplier_name"]))
        for s in run["suppliers"]:
            conn.execute(
                "INSERT INTO supplier_results (rfp_run_id, supplier_name, submission_date, experience_rating,"
                " absolute_score, ppi, industry_index, final_rank, status, result_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RANKED', ?)"
                " ON CONFLICT(rfp_run_id, supplier_name) DO UPDATE SET submission_date=excluded.submission_date,"
                " experience_rating=excluded.experience_rating, absolute_score=excluded.absolute_score,"
                " ppi=excluded.ppi, industry_index=excluded.industry_index, final_rank=excluded.final_rank,"
                " status='RANKED', result_json=excluded.result_json",
                (run_id, s["supplier_name"], s["submission_date"], s["experience_rating"], s["absolute_score"],
                 s["ppi"], s["industry_index"], s["final_rank"], json.dumps(s)),
            )
            for c in s["criteria"]:
                conn.execute(
                    "INSERT INTO criterion_scores (rfp_run_id, supplier_name, criterion_id, criterion_name, weight,"
                    " max_score, score, industry_score, benchmark_score, gap, relative_pct, weighted_points,"
                    " justification, evidence, evidence_verified, overridden)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, s["supplier_name"], c["criterion_id"], c["name"], c["weight"], c["max_score"],
                     c["score"], c.get("industry_score"), c["benchmark_score"], c["gap"], c["relative_pct"],
                     c["weighted_points"], c.get("justification"), c.get("evidence"),
                     int(bool(c.get("evidence_verified"))), int(bool(c.get("overridden")))),
                )
        conn.execute(
            "UPDATE rfp_runs SET status='COMPLETED', warnings_json=?, run_json=?, updated_at=?, revision=?"
            " WHERE rfp_run_id=?",
            (json.dumps(run["warnings"]), json.dumps(run), utc_now(), run.get("revision", 1), run_id),
        )


def list_runs(db_path=None, limit: int = 200) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT r.rfp_run_id, r.created_at, r.status, r.template_name, r.evaluation_mode, r.revision,"
            " (SELECT COUNT(*) FROM supplier_results s WHERE s.rfp_run_id=r.rfp_run_id AND s.status='RANKED') AS suppliers,"
            " (SELECT supplier_name FROM supplier_results s WHERE s.rfp_run_id=r.rfp_run_id AND s.final_rank=1) AS winner"
            " FROM rfp_runs r ORDER BY r.created_at DESC, r.rfp_run_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def load_run(rfp_run_id: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT run_json FROM rfp_runs WHERE rfp_run_id=?", (rfp_run_id,)).fetchone()
        return json.loads(row["run_json"]) if row and row["run_json"] else None


# ------------------------------------------------------------------------------------
# Evaluation DNA (fingerprint of document + criteria + prompt + model)
# ------------------------------------------------------------------------------------


def dna_get(fingerprint: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM evaluation_dna WHERE fingerprint=?", (fingerprint,)).fetchone()
        return dict(r) if r else None


def dna_put(record: dict, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO evaluation_dna (fingerprint, document_sha, criteria_hash, prompt_hash, model,"
            " response, first_run_id, first_supplier, created_at, last_used_at, reuse_count)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (record["fingerprint"], record["document_sha"], record["criteria_hash"], record["prompt_hash"],
             record["model"], record["response"], record.get("first_run_id"), record.get("first_supplier"),
             utc_now(), utc_now()),
        )


def dna_touch(fingerprint: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE evaluation_dna SET reuse_count = reuse_count + 1, last_used_at=? WHERE fingerprint=?",
                     (utc_now(), fingerprint))


def list_dna(db_path=None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT fingerprint, document_sha, model, first_run_id, first_supplier, created_at, last_used_at,"
            " reuse_count FROM evaluation_dna ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def clear_dna(db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM evaluation_dna")


# ------------------------------------------------------------------------------------
# Analyst overrides
# ------------------------------------------------------------------------------------


def add_override(rfp_run_id: str, supplier_name: str, criterion_id: int, criterion_name: str,
                 old_score: float, new_score: float, reason: str, analyst: str = "", db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO score_overrides (rfp_run_id, supplier_name, criterion_id, criterion_name, old_score,"
            " new_score, reason, analyst, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rfp_run_id, supplier_name, criterion_id, criterion_name, old_score, new_score, reason, analyst, utc_now()),
        )


def list_overrides(rfp_run_id: str, db_path=None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM score_overrides WHERE rfp_run_id=? ORDER BY id", (rfp_run_id,)).fetchall()
        return [dict(r) for r in rows]
