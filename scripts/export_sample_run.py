"""Run the 4 sample proposals through the workflow and save the JSON export.

Usage:  python scripts/export_sample_run.py            # offline demo evaluator
        OPENROUTER_API_KEY=... python scripts/export_sample_run.py --llm
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rfp import config, db  # noqa: E402
from rfp.evaluation_agent import LLMEvaluator, OfflineDemoEvaluator  # noqa: E402
from rfp.orchestrator import run_evaluation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--llm", action="store_true", help="use OpenRouter instead of the offline evaluator")
args = parser.parse_args()

meta = json.loads((ROOT / "sample_data" / "sample_metadata.json").read_text())
inputs = [{**m, "filename": f, "file_bytes": (ROOT / "sample_data" / "supplier_pdfs" / f).read_bytes()}
          for f, m in meta.items()]
db.init_db()
if args.llm:
    key = config.get_setting("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY first.")
    evaluator = LLMEvaluator("OpenRouter", key)
else:
    evaluator = OfflineDemoEvaluator()
state = run_evaluation(inputs, evaluator, "KML/HR/2026/014 - HR Service Delivery Platform")
out = ROOT / "sample_output" / ("sample_run_llm.json" if args.llm else "sample_run_offline_demo.json")
out.write_text(json.dumps(state["result"], indent=2))
print(f"Saved {out.relative_to(ROOT)}")
for s in state["result"]["suppliers"]:
    print(f"  #{s['final_rank']} {s['supplier_name']:<16} absolute {s['absolute_score']:6.2f}  PPI {s['ppi']:6.2f}")
