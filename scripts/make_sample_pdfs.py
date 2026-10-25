"""Generate four fictional supplier RFP responses (plus edge-case files).

All companies, people, prices and references are invented for the classroom
project. Each proposal answers the same RFP but with deliberately different
strengths, weaknesses, prices, schedules, evidence quality and gaps:

  Apex Systems     strong technical design + security; highest price; moderate schedule
  BrightPath Tech  lowest price + fastest timeline; weak compliance detail; limited experience
  NexaWorks        balanced; strongest implementation plan and support model
  Orbit Digital    strongest experience + references; vague integration plan; medium price

Usage:  python scripts/make_sample_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data" / "supplier_pdfs"
EDGE = ROOT / "sample_data" / "edge_cases"

RFP_REF = "RFP No. KML/HR/2026/014 - HR Service Delivery Platform"
BUYER = "Kestrel Manufacturing Ltd (fictional)"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=18, leading=22, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#555555"), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1F3A5F"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.8, leading=13.4, spaceAfter=5)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=9, leading=11.5, spaceAfter=0)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=10) for i in items], bulletType="bullet",
                        start="-", leftIndent=12, bulletFontSize=9)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")))
    t.setStyle(TableStyle(style))
    return t


def build(filename: str, supplier: str, date: str, sections: list):
    OUT.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=f"{supplier} - RFP Response",
                            author=supplier)
    story = [
        Paragraph(f"{supplier}: Response to {RFP_REF}", H1),
        Paragraph(f"Submitted to {BUYER} | Submission date: {date} | Fictional proposal for classroom use", SUB),
    ]
    for heading, blocks in sections:
        story.append(Paragraph(heading, H2))
        for b in blocks:
            if isinstance(b, str):
                story.append(Paragraph(b, BODY))
            else:
                story.append(b)
                story.append(Spacer(1, 4))

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(18 * mm, 9 * mm, f"{supplier} | {RFP_REF} | Confidential (fictional)")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


REQUIREMENT = ("Kestrel requires a cloud HR service delivery platform for about 6,500 employees across 4 plants and "
               "a head office: an employee self-service portal, HR case management with SLAs, a policy knowledge "
               "base, and integration with the existing SAP SuccessFactors HRIS and Microsoft Entra ID single sign-on.")


# ---------------------------------------------------------------------------------------------
# 1. APEX SYSTEMS - strong technology and security, premium price, moderate schedule
# ---------------------------------------------------------------------------------------------
def apex():
    build("Apex_Systems_RFP_Response.pdf", "Apex Systems", "2026-10-06", [
        ("1. Executive Summary", [
            "Apex Systems proposes <b>ApexHR Nexus</b>, an enterprise-grade HR service delivery platform built on a "
            "cloud-native microservices architecture. Our design prioritises integration depth, resilience and "
            "security, and is engineered to scale well beyond Kestrel's current 6,500 employees.",
            "We commit to a contractual availability of 99.95% with active-active deployment across two Indian "
            "Azure regions (Pune and Chennai), and we hold ISO 27001:2022 and SOC 2 Type II certifications.",
        ]),
        ("2. Our Understanding of the Requirement", [
            REQUIREMENT,
            "We understand the key success factors to be: accurate employee data from SuccessFactors, a single "
            "front door for employee queries, measurable SLA adherence, and auditable handling of sensitive HR data.",
        ]),
        ("3. Proposed Solution and Architecture", [
            "ApexHR Nexus is composed of independently deployable services (Portal, Case Engine, Knowledge, "
            "Workflow, Analytics) running on Kubernetes (AKS) behind an API gateway.",
            bullets([
                "<b>HRIS integration:</b> certified SAP SuccessFactors connector using OData v4 APIs with near real-time "
                "delta sync every 15 minutes and event-driven updates for joiners, movers and leavers.",
                "<b>Identity:</b> SAML 2.0 / OIDC single sign-on with Microsoft Entra ID; SCIM 2.0 user provisioning.",
                "<b>Open APIs:</b> 140+ documented REST APIs and webhooks for payroll, ITSM (ServiceNow) and Teams.",
                "<b>Scalability:</b> load tested to 50,000 concurrent users with p95 page response under 1.2 seconds.",
                "<b>Data model:</b> configurable case categories, dynamic forms and a rules engine without code changes.",
                "<b>Analytics:</b> embedded dashboards for case volumes, SLA breach risk and knowledge-article deflection.",
            ]),
        ]),
        ("4. Implementation Approach, Timeline and Team", [
            "We follow a four-phase approach over <b>18 weeks</b>: Discover and Design (weeks 1-4), Build and "
            "Integrate (weeks 5-11), Test and Train (weeks 12-15) and Go-live with Hypercare (weeks 16-18).",
            table([
                ["Milestone", "Week", "Exit criteria"],
                ["Solution design sign-off", "4", "Approved design, integration specification"],
                ["SuccessFactors and SSO integration complete", "10", "Integration test pass rate 98%+"],
                ["User acceptance testing complete", "15", "No open Severity 1 or 2 defects"],
                ["Go-live (all sites)", "16", "Cutover checklist signed"],
            ], [70, 18, 86]),
            "Team: engagement director, solution architect, 2 integration engineers, 2 configuration consultants, "
            "QA lead and change-management consultant. The detailed risk register will be produced during discovery.",
        ]),
        ("5. Commercial Proposal", [
            table([
                ["Item", "Year 1 (INR)", "Year 2 (INR)", "Year 3 (INR)"],
                ["Subscription (6,500 employees)", "58,50,000", "58,50,000", "61,42,500"],
                ["Implementation services (fixed price)", "52,00,000", "-", "-"],
                ["Premium support and TAM", "6,00,000", "6,00,000", "6,00,000"],
                ["<b>Total</b>", "<b>1,16,50,000</b>", "<b>64,50,000</b>", "<b>67,42,500</b>"],
            ], [68, 34, 34, 34]),
            "Three-year total cost of ownership: <b>INR 2,48,42,500</b> excluding GST. Assumptions: prices are valid "
            "for 90 days; subscription uplift of 5% in Year 3; customer provides SuccessFactors API access and test "
            "tenants; travel billed at actuals capped at 3% of services.",
        ]),
        ("6. Security, Compliance and Risk Controls", [
            bullets([
                "ISO 27001:2022 certified (certificate no. AX-ISMS-4471, valid to March 2028) and SOC 2 Type II report "
                "for the period April 2025 - March 2026 available under NDA.",
                "Encryption: AES-256 at rest with customer-managed keys in Azure Key Vault; TLS 1.3 in transit.",
                "India data residency; compliant with the Digital Personal Data Protection Act 2023, with consent and "
                "data-principal request workflows built in.",
                "Role-based and attribute-based access control; MFA enforced for all administrators.",
                "Immutable audit logs retained for 7 years and exportable to the customer SIEM.",
                "Independent VAPT twice a year by a CERT-In empanelled auditor; critical findings fixed within 7 days.",
            ]),
        ]),
        ("7. Support Model and Relevant Experience", [
            "Support is provided 24x7 for Priority 1 incidents (response within 30 minutes) and business hours for "
            "lower priorities, with a named Technical Account Manager.",
            "Apex has delivered HR service delivery platforms for 9 enterprises in the last six years, including a "
            "12,000-employee automotive components manufacturer (fictional reference: Meridian Auto Parts, HR "
            "Director available on request).",
        ]),
    ])


# ---------------------------------------------------------------------------------------------
# 2. BRIGHTPATH TECH - lowest price, fastest, weak compliance, limited experience
# ---------------------------------------------------------------------------------------------
def brightpath():
    build("BrightPath_Tech_RFP_Response.pdf", "BrightPath Tech", "2026-10-03", [
        ("1. Executive Summary", [
            "BrightPath Tech is pleased to offer <b>BrightDesk HR</b>, a modern and affordable HR helpdesk and "
            "employee portal. We can get Kestrel live in just <b>10 weeks</b> at the lowest total cost in the market.",
            "As a young and agile company founded in 2023, we bring energy, flexibility and a startup mindset.",
        ]),
        ("2. Understanding of Requirements", [
            "We understand that Kestrel needs an employee portal, case management and a knowledge base for its "
            "employees, with connection to its HR system and login.",
        ]),
        ("3. Proposed Solution", [
            "BrightDesk HR is a single multi-tenant SaaS application. Employees can raise tickets through the "
            "portal, email or a chatbot, and HR agents manage them in a shared queue with SLA timers.",
            bullets([
                "Ready-made templates for common HR queries (payslips, leave, letters).",
                "Knowledge base with article search.",
                "Integration with HR systems through CSV file upload, with an API connector planned for 2027.",
                "Login with Google or Microsoft accounts.",
                "Mobile-friendly web interface.",
            ]),
        ]),
        ("4. Implementation Plan", [
            "Our rapid deployment method gets customers live in 10 weeks:",
            table([
                ["Weeks", "Activity"],
                ["1-2", "Kick-off and configuration workshops"],
                ["3-7", "Configuration and data upload"],
                ["8-9", "Testing and training"],
                ["10", "Go-live"],
            ], [30, 140]),
            "A project manager and two consultants will be assigned. Testing will be done by Kestrel's HR team.",
        ]),
        ("5. Pricing", [
            table([
                ["Item", "Cost (INR)"],
                ["Annual subscription (per year, up to 7,000 employees)", "29,00,000"],
                ["One-time implementation", "14,00,000"],
                ["<b>3-year total</b>", "<b>1,01,00,000</b>"],
            ], [120, 50]),
            "Support is included in the subscription. Prices exclude taxes. Additional customisation will be quoted "
            "separately on a time-and-materials basis at INR 4,500 per hour.",
        ]),
        ("6. Security", [
            "Security is very important to us. BrightDesk is hosted on a leading public cloud and we follow industry "
            "best practices to keep customer data safe. Data is encrypted. We are currently working towards ISO "
            "certification. Further details on security can be shared after contract signing.",
        ]),
        ("7. Support and Experience", [
            "Our support team is available on email from 10am to 6pm, Monday to Friday. We currently serve 3 "
            "customers in the services sector, the largest with 900 employees. References are not available at this "
            "stage due to confidentiality agreements.",
        ]),
    ])


# ---------------------------------------------------------------------------------------------
# 3. NEXAWORKS - balanced; strongest implementation plan and support model
# ---------------------------------------------------------------------------------------------
def nexaworks():
    build("NexaWorks_RFP_Response.pdf", "NexaWorks", "2026-10-07", [
        ("1. Executive Summary", [
            "NexaWorks proposes <b>NexaServe HR</b>, a configurable HR service delivery platform, together with a "
            "delivery and support model designed around manufacturing workforces with shift-based, frontline and "
            "office employees. Our proposal balances proven technology, a realistic fixed-price plan and a support "
            "model with financially backed service levels.",
        ]),
        ("2. Our Understanding", [
            REQUIREMENT,
            "Frontline employees at the plants often lack corporate laptops, so we include kiosk and WhatsApp access "
            "and Hindi, Marathi and Tamil language packs.",
        ]),
        ("3. Proposed Solution", [
            bullets([
                "Employee portal, mobile app and kiosk mode with multilingual knowledge base.",
                "HR case management with SLA timers, escalation matrix and confidential case handling for "
                "employee-relations matters.",
                "Pre-built SAP SuccessFactors integration using the standard OData APIs (daily full sync plus "
                "hourly delta), and single sign-on via Microsoft Entra ID (SAML 2.0).",
                "REST APIs for payroll and the ServiceNow IT service desk.",
                "Hosted on AWS Mumbai region with 99.9% availability commitment.",
            ]),
        ]),
        ("4. Implementation Plan, Team and Milestones", [
            "We use a phased, fixed-price delivery over <b>16 weeks</b> with a pilot plant before full roll-out. "
            "Governance includes a weekly working group and a fortnightly steering committee chaired by Kestrel's "
            "CHRO.",
            table([
                ["Phase", "Weeks", "Key deliverables", "Exit criteria"],
                ["1. Mobilise and discover", "1-3", "Process maps, integration design, RAID log", "Design sign-off"],
                ["2. Configure and integrate", "4-9", "Configured portal, SuccessFactors and SSO integration",
                 "System test pass rate 95%+"],
                ["3. Pilot (Pune plant)", "10-12", "Pilot with 800 employees, feedback fixes",
                 "Pilot CSAT 4/5 or higher"],
                ["4. Roll-out (3 plants + HO)", "13-15", "Wave go-lives, trained HR agents", "Cutover checklist signed"],
                ["5. Hypercare", "16 (+4 wks)", "Daily stand-ups, stabilisation", "Steady-state handover"],
            ], [36, 18, 66, 50]),
            "<b>Named team:</b> delivery lead (PMP certified, 14 years), solution architect, integration engineer, "
            "two functional consultants, test lead, and a change and training lead. A RACI matrix and responsibility "
            "split with Kestrel is included in Annexure A.",
            "<b>Top risks and mitigations:</b> (1) SuccessFactors data quality - data-cleansing sprint in weeks 2-4; "
            "(2) low frontline adoption - plant champions network and kiosk access; (3) HR agent capacity during "
            "go-live - phased waves and floor-walkers; (4) scope creep - change-control board with documented "
            "approval thresholds.",
        ]),
        ("5. Commercial Proposal", [
            table([
                ["Item", "Year 1 (INR)", "Year 2 (INR)", "Year 3 (INR)"],
                ["Subscription (6,500 employees)", "44,00,000", "44,00,000", "44,00,000"],
                ["Implementation (fixed price, milestone-based)", "30,00,000", "-", "-"],
                ["Enhanced support (included)", "0", "0", "0"],
                ["<b>Total</b>", "<b>74,00,000</b>", "<b>44,00,000</b>", "<b>44,00,000</b>"],
            ], [70, 33, 33, 33]),
            "Three-year total: <b>INR 1,62,00,000</b> excluding GST. Subscription price is locked for three years. "
            "Payment for implementation is linked to the five phase exit criteria. Assumption: Kestrel nominates "
            "process owners for each HR service line.",
        ]),
        ("6. Security and Compliance", [
            "NexaWorks is ISO 27001 certified and completed a SOC 2 Type I audit in 2025 (Type II in progress). "
            "Data is encrypted at rest and in transit, hosted in India, and processed in line with the DPDP Act "
            "2023. Access is role-based with MFA for administrators, and audit logs are retained for 3 years. "
            "Annual penetration testing is performed by an independent firm.",
        ]),
        ("7. Support Model and Experience", [
            table([
                ["Priority", "Response", "Resolution target", "Service credit if missed"],
                ["P1 - platform down", "15 minutes, 24x7", "4 hours", "10% of monthly fee"],
                ["P2 - major function impaired", "1 hour, 24x7", "8 hours", "5% of monthly fee"],
                ["P3 - minor issue", "4 business hours", "3 business days", "-"],
            ], [44, 38, 40, 48]),
            "A dedicated Customer Success Manager runs monthly service reviews and quarterly adoption health checks. "
            "NexaWorks has completed 11 HR service delivery implementations, including two manufacturers with "
            "multi-plant frontline workforces (fictional references: Sahyadri Steel, 4,800 employees; Coastal "
            "Polymers, 3,200 employees).",
        ]),
    ])


# ---------------------------------------------------------------------------------------------
# 4. ORBIT DIGITAL - strongest experience and references, vague integration plan, medium price
# ---------------------------------------------------------------------------------------------
def orbit():
    build("Orbit_Digital_RFP_Response.pdf", "Orbit Digital", "2026-10-07", [
        ("1. Executive Summary", [
            "Orbit Digital has been delivering HR technology for <b>17 years</b>, with more than <b>40 HR service "
            "delivery implementations</b> across manufacturing, pharmaceuticals and retail in India and South-East "
            "Asia. We propose our <b>OrbitPeople Service Hub</b> together with our proven delivery practice.",
        ]),
        ("2. Our Understanding", [
            REQUIREMENT,
        ]),
        ("3. Proposed Solution", [
            "OrbitPeople Service Hub provides an employee portal, HR case management, knowledge management and "
            "reporting. The platform is used by over 300,000 employees across our client base.",
            "Integration with SuccessFactors and Microsoft Entra ID will be addressed during the discovery phase. "
            "The integration approach (middleware or direct connectors) and data-sync frequency will be finalised "
            "after discovery once we understand Kestrel's landscape. Our team has integrated with many HR systems "
            "in the past.",
        ]),
        ("4. Implementation Approach", [
            "Our delivery methodology, OrbitPath, has been refined over 40+ projects. We expect an overall duration of "
            "around <b>20 weeks</b>, split into discovery (4 weeks), build (9 weeks), testing (5 weeks) and go-live "
            "(2 weeks). A detailed plan with milestones will be shared after discovery. A project manager and a "
            "team of consultants will be deployed as needed.",
        ]),
        ("5. Commercial Proposal", [
            table([
                ["Item", "Amount (INR)"],
                ["Subscription - 3 years (6,500 employees)", "1,23,00,000"],
                ["Implementation services (estimate)", "38,00,000"],
                ["Support (standard, included)", "Included"],
                ["<b>Total 3-year estimate</b>", "<b>1,61,00,000</b>"],
            ], [110, 60]),
            "Implementation services are estimated and may change after discovery. Integration effort is not "
            "included and will be quoted after discovery. Prices exclude GST.",
        ]),
        ("6. Security and Compliance", [
            "Orbit Digital is ISO 27001 certified. Customer data is hosted in India and encrypted. We comply with the "
            "DPDP Act 2023 and conduct annual security audits. Role-based access is available.",
        ]),
        ("7. Support, Experience and References", [
            "Support is available 8am-8pm IST on weekdays through our service desk, with P1 response within 1 hour. "
            "A dedicated account manager is assigned to every enterprise client.",
            table([
                ["Reference client (fictional)", "Industry / size", "Scope", "Contact"],
                ["Vardhan Pharma", "Pharma, 9,000 employees", "HR helpdesk, knowledge base, 5 sites",
                 "Head of HR Operations"],
                ["Shivam Auto Components", "Manufacturing, 7,200", "Portal + case management, 6 plants", "CHRO"],
                ["Lotus Retail", "Retail, 15,000", "Multilingual HR helpdesk, 220 stores", "VP People Services"],
            ], [40, 38, 54, 38]),
            "Client satisfaction across our HR service delivery practice averaged 4.6/5 in our 2025 survey, and 90% "
            "of clients have renewed for a second term.",
        ]),
    ])


def edge_cases():
    """Files used to demonstrate validation of bad inputs."""
    EDGE.mkdir(parents=True, exist_ok=True)
    # (a) Image-only "scanned" PDF: has no extractable text layer.
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(EDGE / "Scanned_Proposal_No_Text.pdf"), pagesize=A4)
    c.setFillColor(colors.HexColor("#DDDDDD"))
    for i in range(12):
        c.rect(25 * mm, (250 - i * 18) * mm, 160 * mm, 6 * mm, fill=1, stroke=0)
    c.save()
    # (b) Not a PDF at all, but named .pdf
    (EDGE / "Corrupt_File.pdf").write_text("This is not really a PDF document.")


if __name__ == "__main__":
    apex(); brightpath(); nexaworks(); orbit(); edge_cases()
    for p in sorted(OUT.glob("*.pdf")) + sorted(EDGE.glob("*.pdf")):
        print(p.relative_to(ROOT))
