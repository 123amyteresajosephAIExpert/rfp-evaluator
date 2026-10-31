"""Report Tool: builds a shareable PDF evaluation report for one run (ReportLab)."""

from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#2A6FDB")
LIGHT = colors.HexColor("#EEF3FA")
GRID = colors.HexColor("#C9D3DF")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Title"], fontSize=20, leading=24, textColor=NAVY, alignment=0, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=13, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=_ss["Heading3"], fontSize=11, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("B", parent=_ss["Normal"], fontSize=9.2, leading=12.5, spaceAfter=3)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#4A5562"))
CELL = ParagraphStyle("C", parent=BODY, fontSize=8, leading=10, spaceAfter=0)


def _p(text, style=BODY):
    return Paragraph(escape(str(text)), style)


def _table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(escape(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def _header_cells(rows):
    rows[0] = [Paragraph(f"<b><font color='white'>{escape(str(c))}</font></b>", CELL) for c in rows[0]]
    return rows


def _ppi_chart(suppliers):
    n = len(suppliers)
    h = 26 + 22 * n
    d = Drawing(170 * mm, h)
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 40 * mm, 10, 120 * mm, h - 20
    ordered = list(reversed(suppliers))
    ch.data = [[s["ppi"] for s in ordered]]
    ch.categoryAxis.categoryNames = [s["supplier_name"] for s in ordered]
    ch.categoryAxis.labels.fontSize = 8
    ch.valueAxis.valueMin, ch.valueAxis.valueMax, ch.valueAxis.valueStep = 0, 100, 20
    ch.valueAxis.labels.fontSize = 7
    ch.bars[0].fillColor = BLUE
    ch.bars[0].strokeColor = None
    ch.barWidth = 10
    d.add(ch)
    for i, s in enumerate(ordered):
        y = ch.y + (i + 0.5) * ch.height / n - 3
        d.add(String(ch.x + ch.width * s["ppi"] / 100 + 3, y, f"{s['ppi']:.1f}", fontSize=7.5))
    return d


def build_run_report(run: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=f"Evaluation report {run['rfp_run_id']}")
    sups = run["suppliers"]
    tmpl = run.get("template", {})
    story = [
        _p("RFP Evaluation Report", H1),
        _p(f"{run.get('rfp_title', '')}  |  Run {run['rfp_run_id']}  |  revision {run.get('revision', 1)}  |  "
           f"created {run['created_at']}  |  engine {run['evaluation_mode']}", SMALL),
        Spacer(1, 6),
    ]
    if sups:
        top = sups[0]
        story.append(_p(f"Recommended supplier: {top['supplier_name']} - PPI {top['ppi']:.2f}, absolute score "
                        f"{top['absolute_score']:.2f}/100, industry alignment {top['industry_index']:.1f}/100.", H3))
    if tmpl.get("summary"):
        story.append(_p(f"Requirement: {tmpl['summary']}"))

    story.append(_p("Leaderboard", H2))
    rows = [["Rank", "Supplier", "PPI", "Absolute /100", "Industry /100", "Submitted", "Exp. rating"]]
    rows += [[str(s["final_rank"]), s["supplier_name"], f"{s['ppi']:.2f}", f"{s['absolute_score']:.2f}",
              f"{s['industry_index']:.1f}", s["submission_date"], f"{s['experience_rating']:g}"] for s in sups]
    story += [_table(_header_cells(rows), [12, 48, 20, 24, 24, 26, 20]), Spacer(1, 6), _ppi_chart(sups)]
    story.append(_p("Ranking rule: higher PPI, then earlier submission date, then higher experience rating, then "
                    "supplier name A-Z. PPI = weighted average of each criterion score as a % of the best peer score.", SMALL))

    story.append(_p("Criteria and weights", H2))
    rows = [["Criterion", "Weight", "Max", "What was inspected", "Industry best practice"]]
    rows += [[c["name"], f"{c['weight']:g}%", str(c["max_score"]), c["description"], c.get("industry_benchmark", "")]
             for c in run["criteria"]]
    story.append(_table(_header_cells(rows), [32, 15, 10, 58, 63]))

    story.append(_p("Score matrix (score out of max; benchmark = best peer score)", H2))
    head = ["Supplier"] + [c["name"] for c in run["criteria"]]
    rows = [head] + [[s["supplier_name"]] + [f"{c['score']:g} (rel {c['relative_pct']:.0f}%)" for c in s["criteria"]]
                     for s in sups]
    w = 178 / len(head)
    story.append(_table(_header_cells(rows), [w] * len(head)))

    for s in sups:
        story.append(PageBreak())
        story.append(_p(f"#{s['final_rank']}  {s['supplier_name']}", H2))
        story.append(_p(f"PPI {s['ppi']:.2f}  |  Absolute {s['absolute_score']:.2f}/100  |  Industry alignment "
                        f"{s['industry_index']:.1f}/100  |  Source file {s.get('source_file', '')}", SMALL))
        if s.get("overall_summary"):
            story.append(_p(s["overall_summary"]))
        if s.get("rank_explanation"):
            story.append(_p(s["rank_explanation"], SMALL))
        for label, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"), ("Risks", "risks")):
            if s.get(key):
                story.append(Paragraph(f"<b>{label}:</b> " + escape("; ".join(s[key])), BODY))
        rows = [["Criterion", "Score", "Bench.", "Gap", "Rel. %", "Industry", "Evidence (verified?) and justification"]]
        for c in s["criteria"]:
            ev = c.get("evidence") or "none"
            flag = "verified" if c.get("evidence_verified") else "NOT verified"
            over = f" [overridden from {c['original_score']:g}: {c.get('override_reason', '')}]" if c.get("overridden") else ""
            rows.append([c["name"], f"{c['score']:g}/{c['max_score']}{over}", f"{c['benchmark_score']:g}",
                         f"{c['gap']:g}", f"{c['relative_pct']:.0f}", f"{c['industry_score']:g}",
                         f"\"{ev}\" ({flag}) - {c.get('justification', '')}"])
        story.append(_table(_header_cells(rows), [28, 20, 12, 10, 12, 14, 82]))

    story.append(PageBreak())
    story.append(_p("Audit trail", H2))
    rows = [["Position", "Deciding rule", "Explanation"]]
    rows += [[f"#{t['upper_rank']} {t['upper']} above #{t['lower_rank']} {t['lower']}", t["deciding_rule"],
              t["explanation"]] for t in run["tie_breaks"]]
    if len(rows) > 1:
        story += [_p("Tie-break decisions", H3), _table(_header_cells(rows), [55, 35, 88])]
    if run.get("overrides"):
        rows = [["Supplier", "Criterion", "Change", "Reason", "When"]]
        rows += [[o["supplier_name"], o["criterion_name"], f"{o['old_score']:g} -> {o['new_score']:g}", o["reason"],
                  o["created_at"]] for o in run["overrides"]]
        story += [_p("Analyst overrides", H3), _table(_header_cells(rows), [32, 32, 20, 60, 34])]
    if run.get("warnings"):
        story.append(_p(f"Validation warnings ({len(run['warnings'])})", H3))
        story += [_p(f"- {w}", SMALL) for w in run["warnings"][:60]]
    if run.get("not_evaluated"):
        story.append(_p("Files not evaluated", H3))
        story += [_p(f"- {n['supplier_name']} ({n['filename']}): {n['reason']}", SMALL) for n in run["not_evaluated"]]
    story.append(_p("Formulas", H3))
    rows = [["Metric", "Formula"]] + [[k, v] for k, v in run.get("formulas", {}).items()]
    story.append(KeepTogether(_table(_header_cells(rows), [55, 123])))
    story.append(Spacer(1, 6))
    story.append(_p("The AI judged proposal content only. All totals, benchmarks, tie-breaks and ranks were computed "
                    "by deterministic code. This report supports, and does not replace, the procurement decision.", SMALL))

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(16 * mm, 8 * mm, f"Agentic RFP Evaluator | {run['rfp_run_id']}")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
