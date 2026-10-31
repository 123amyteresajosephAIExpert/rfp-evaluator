"""Create the SQLite database and seed the sample template and criteria.

Usage:  python scripts/seed_db.py            # create if missing (idempotent)
        python scripts/seed_db.py --reset    # delete and recreate the database
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfp import config, db  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="delete the existing database first")
args = parser.parse_args()
if args.reset and config.DB_PATH.exists():
    config.DB_PATH.unlink()
    print(f"Deleted {config.DB_PATH}")
db.init_db()
print(f"Database ready at {config.DB_PATH}")
for t in db.list_templates():
    print(f"Template [{t['template_id']}] {t['name']}")
    for c in db.get_criteria(t["template_id"], active_only=False):
        print(f"  - {c['name']:<26} weight {c['weight']:>5g}%  max {c['max_score']}  active={c['is_active']}")
