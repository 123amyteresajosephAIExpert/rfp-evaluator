"""Run the 5 sample HR proposals through the workflow (offline evaluator) and save the JSON export.

Usage:  python scripts/export_sample_run.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rfp import db  # noqa: E402
from rfp.evaluation_agent import OfflineDemoEvaluator  # noqa: E402
from rfp.orchestrator import run_evaluation  # noqa: E402

meta = json.loads((ROOT / "sample_data" / "sample_metadata.json").read_text())
folder = ROOT / "sample_data" / "proposals" / "hr_platform"
inputs = [{**meta[p.name], "filename": p.name, "file_bytes": p.read_bytes()} for p in sorted(folder.glob("*.pdf"))]
db.init_db()
state = run_evaluation(1, inputs, OfflineDemoEvaluator())
out = ROOT / "sample_output" / "sample_run_offline_demo.json"
out.write_text(json.dumps(state["result"], indent=2))
print(f"Saved {out.relative_to(ROOT)}")
for s in state["result"]["suppliers"]:
    print(f"  #{s['final_rank']} {s['supplier_name']:<22} PPI {s['ppi']:6.2f}  absolute {s['absolute_score']:6.2f}")
