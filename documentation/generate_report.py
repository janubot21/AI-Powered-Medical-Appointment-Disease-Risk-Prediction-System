from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DOCS_DIR = ROOT / "documentation"
OUTPUT_PDF = DOCS_DIR / "AI_Powered_Medical_Appointment_and_Disease_Risk_Prediction_System_Report.pdf"

PROJECT_TITLE = "AI-Powered Medical Appointment and Disease Risk Prediction System"
STUDENT_NAME = "Student Name"
ROLL_NUMBER = "Roll Number"
GUIDE_NAME = "Guide Name"
GUIDE_DESIGNATION = "Project Guide"
UNIVERSITY_NAME = "University Name"
DEPARTMENT_NAME = "Department of Computer Science"
ACADEMIC_YEAR = "2025-2026"


@dataclass
class ProjectFacts:
    clean_rows: int
    clean_cols: int
    feature_rows: int
    feature_cols: int
    model_rows: int
    dropped_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    risk_distribution: dict[str, int]
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix_text: str
    portal_patient_accounts: int
    api_patient_accounts: int
    doctor_accounts: int
    nurse_accounts: int
    doctor_profiles: int
    emergency_doctors: int
    doctor_leaves: int
    new_patient_rows: int
    profile_rows: int
    appointment_rows: int
    template_count: int
    static_asset_count: int


class NumberedCanvasDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self.allowSplitting = 1
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._on_page)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = flowable.getPlainText()
            if style_name == "Heading1":
                self.notify("TOCEntry", (0, text, self.page))
            elif style_name == "Heading2":
                self.notify("TOCEntry", (1, text, self.page))

    def _on_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 10)
        canvas.drawCentredString(A4[0] / 2.0, 1.4 * cm, f"Page {doc.page}")
        canvas.restoreState()


def derive_risk_level(row: pd.Series) -> str | None:
    age = row.get("Age")
    symptom_count = row.get("Symptom_Count")
    if pd.isna(age) or pd.isna(symptom_count):
        return None

    score = 0
    if age > 60:
        score += 2
    elif age > 40:
        score += 1

    if symptom_count >= 6:
        score += 2
    elif symptom_count >= 3:
        score += 1

    if score <= 2:
        return "Low"
    if score <= 4:
        return "Medium"
    return "High"


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def csv_truthy_count(path: Path, column_name: str, *, truthy_values: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if column_name not in df.columns:
        return 0
    truthy = truthy_values or {"1", "true", "yes", "y"}
    return int(df[column_name].astype(str).str.strip().str.lower().isin(truthy).sum())


def directory_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def sqlite_table_count(db_path: Path, table_name: str) -> int:
    import sqlite3

    if not db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return 0


def gather_project_facts() -> ProjectFacts:
    clean_df = pd.read_csv(BACKEND / "Healthcare_Cleaned.csv")
    feature_df = pd.read_csv(BACKEND / "Healthcare_FeatureEngineered.csv")
    if "Risk_Level" not in feature_df.columns:
        feature_df["Risk_Level"] = feature_df.apply(derive_risk_level, axis=1)

    feature_cols_df = feature_df.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")
    model_df = feature_df.dropna(subset=list(feature_cols_df.columns) + ["Risk_Level"]).copy()

    x_data = model_df.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")
    y_data = model_df["Risk_Level"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = joblib.load(BACKEND / "decision_tree_model.pkl")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100.0
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    risk_distribution = dict(Counter(model_df["Risk_Level"]))

    datasets_dir = BACKEND / "datasets"
    templates_dir = ROOT / "frontend" / "templates"
    static_dir = ROOT / "frontend" / "static"
    return ProjectFacts(
        clean_rows=len(clean_df),
        clean_cols=len(clean_df.columns),
        feature_rows=len(feature_df),
        feature_cols=len(feature_df.columns),
        model_rows=len(model_df),
        dropped_rows=len(feature_df) - len(model_df),
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_count=len(getattr(model, "feature_names_in_", [])),
        risk_distribution=risk_distribution,
        test_accuracy=round(accuracy, 2),
        precision=round(precision * 100.0, 2),
        recall=round(recall * 100.0, 2),
        f1_score=round(f1_score * 100.0, 2),
        confusion_matrix_text=str(cm.tolist()),
        portal_patient_accounts=csv_row_count(datasets_dir / "patients.csv"),
        api_patient_accounts=csv_row_count(datasets_dir / "patient_accounts.csv"),
        doctor_accounts=csv_row_count(datasets_dir / "doctor_accounts.csv"),
        nurse_accounts=csv_row_count(datasets_dir / "nurse_accounts.csv"),
        doctor_profiles=csv_row_count(datasets_dir / "doctor_profiles.csv"),
        emergency_doctors=csv_truthy_count(datasets_dir / "doctor_profiles.csv", "emergency_doctor"),
        doctor_leaves=csv_row_count(datasets_dir / "doctor_leave.csv"),
        new_patient_rows=csv_row_count(datasets_dir / "new_patient_data.csv"),
        profile_rows=sqlite_table_count(datasets_dir / "patient_records.db", "patient_profiles"),
        appointment_rows=sqlite_table_count(datasets_dir / "patient_records.db", "appointments"),
        template_count=directory_file_count(templates_dir),
        static_asset_count=directory_file_count(static_dir),
    )


def build_styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Times-Bold"
    styles["Title"].fontSize = 20
    styles["Title"].leading = 26
    styles["Title"].alignment = TA_CENTER

    styles.add(
        ParagraphStyle(
            name="CoverLine",
            fontName="Times-Roman",
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyLeft",
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles["Heading1"].fontName = "Times-Bold"
    styles["Heading1"].fontSize = 15
    styles["Heading1"].leading = 20
    styles["Heading1"].spaceBefore = 10
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].fontName = "Times-Bold"
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 6
    return styles


def bullet_points(items: Iterable[str], style) -> list[Paragraph]:
    return [Paragraph(f"&#8226; {item}", style) for item in items]


def styled_table(rows, col_widths=None):
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_architecture_diagram() -> Drawing:
    drawing = Drawing(460, 240)
    box_fill = colors.HexColor("#EAF2F8")
    stroke = colors.HexColor("#4F81BD")
    boxes = [
        (20, 175, 120, 40, "Patient Portal"),
        (20, 110, 120, 40, "Doctor Portal"),
        (20, 45, 120, 40, "Nurse Portal"),
        (165, 135, 120, 60, "FastAPI Entry +\nCompatibility Routes"),
        (165, 55, 120, 55, "Flask Portal +\nSession Workflows"),
        (315, 190, 125, 35, "Risk Engine\nDecision Tree"),
        (315, 140, 125, 35, "Triage + Leave /\nEmergency Rules"),
        (315, 90, 125, 35, "OpenAI Patient\nChat (Optional)"),
        (315, 35, 125, 35, "CSV + SQLite\nStorage"),
    ]
    for x_pos, y_pos, width, height, label in boxes:
        drawing.add(Rect(x_pos, y_pos, width, height, fillColor=box_fill, strokeColor=stroke, strokeWidth=1.2))
        lines = label.split("\n")
        base_y = y_pos + height / 2 + (6 if len(lines) == 1 else 12)
        for index, line in enumerate(lines):
            drawing.add(String(x_pos + width / 2, base_y - (index * 12), line, fontName="Times-Roman", fontSize=10, textAnchor="middle"))
    arrows = [
        ((140, 195), (165, 175)),
        ((140, 130), (165, 165)),
        ((140, 65), (165, 85)),
        ((140, 195), (165, 92)),
        ((140, 130), (165, 92)),
        ((140, 65), (165, 82)),
        ((285, 175), (315, 207)),
        ((285, 165), (315, 157)),
        ((285, 82), (315, 107)),
        ((285, 82), (315, 52)),
    ]
    for start, end in arrows:
        drawing.add(Line(start[0], start[1], end[0], end[1], strokeColor=colors.black, strokeWidth=1))
    return drawing


def build_flow_diagram() -> Drawing:
    drawing = Drawing(460, 205)
    fill = colors.HexColor("#FDF2E9")
    stroke = colors.HexColor("#C55A11")
    labels = [
        (15, 120, 85, 40, "Patient\nLogin"),
        (115, 120, 90, 40, "Health Data\nSubmission"),
        (220, 120, 95, 40, "Risk + Chat\nSummary"),
        (330, 120, 100, 40, "Availability /\nLeave Check"),
        (330, 50, 100, 40, "Book or Cancel\nAppointment"),
        (115, 50, 90, 40, "Nurse Record\nUpdate"),
        (220, 50, 95, 40, "Doctor Review\nDashboard"),
    ]
    for x_pos, y_pos, width, height, label in labels:
        drawing.add(Rect(x_pos, y_pos, width, height, fillColor=fill, strokeColor=stroke, strokeWidth=1.2))
        for index, line in enumerate(label.split("\n")):
            drawing.add(String(x_pos + width / 2, y_pos + height / 2 + 6 - (index * 12), line, fontName="Times-Roman", fontSize=10, textAnchor="middle"))
    connectors = [
        ((100, 140), (115, 140)),
        ((205, 140), (220, 140)),
        ((315, 140), (330, 140)),
        ((380, 120), (380, 90)),
        ((205, 70), (220, 70)),
        ((160, 120), (160, 90)),
    ]
    for start, end in connectors:
        drawing.add(Line(start[0], start[1], end[0], end[1], strokeColor=colors.black, strokeWidth=1))
    return drawing


def figure_with_caption(image_path: Path, caption: str, styles, width_cm: float = 13.0, max_height_cm: float = 7.2):
    img = Image(str(image_path))
    img.drawWidth = width_cm * cm
    img.drawHeight = img.drawHeight * (img.drawWidth / img.imageWidth)
    max_height = max_height_cm * cm
    if img.drawHeight > max_height:
        scale = max_height / img.drawHeight
        img.drawHeight = max_height
        img.drawWidth = img.drawWidth * scale
    return [img, Spacer(1, 0.15 * cm), Paragraph(f"<i>{caption}</i>", styles["Small"]), Spacer(1, 0.25 * cm)]


def build_story(facts: ProjectFacts):
    styles = build_styles()
    story = []

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOCHeading1", fontName="Times-Roman", fontSize=11, leftIndent=18, firstLineIndent=-10, spaceBefore=4),
        ParagraphStyle(name="TOCHeading2", fontName="Times-Roman", fontSize=10, leftIndent=36, firstLineIndent=-10, spaceBefore=2),
    ]

    story.extend(
        [
            Spacer(1, 2 * cm),
            Paragraph(UNIVERSITY_NAME, styles["Title"]),
            Spacer(1, 0.5 * cm),
            Paragraph(DEPARTMENT_NAME, styles["CoverLine"]),
            Spacer(1, 1.1 * cm),
            Table(
                [[Paragraph(PROJECT_TITLE, styles["Title"])]],
                colWidths=[15.5 * cm],
                style=TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 1.2, colors.black),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 18),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
                    ]
                ),
                hAlign="CENTER",
            ),
            Spacer(1, 1.2 * cm),
            Paragraph("Project Documentation Report", styles["CoverLine"]),
            Spacer(1, 1.2 * cm),
            Paragraph(f"<b>Submitted By:</b> {STUDENT_NAME}", styles["CoverLine"]),
            Paragraph(f"<b>Roll Number:</b> {ROLL_NUMBER}", styles["CoverLine"]),
            Spacer(1, 0.7 * cm),
            Paragraph(f"<b>Guided By:</b> {GUIDE_NAME}", styles["CoverLine"]),
            Paragraph(f"<b>Designation:</b> {GUIDE_DESIGNATION}", styles["CoverLine"]),
            Spacer(1, 1.3 * cm),
            Paragraph(f"<b>Academic Year:</b> {ACADEMIC_YEAR}", styles["CoverLine"]),
            Spacer(1, 2.5 * cm),
            Paragraph("Submitted in partial fulfillment of the requirements for the degree program.", styles["CoverLine"]),
            PageBreak(),
        ]
    )

    story.append(Paragraph("Declaration by the Student", styles["Heading1"]))
    story.append(
        Paragraph(
            "I hereby declare that the project work entitled "
            f'"{PROJECT_TITLE}" submitted to {UNIVERSITY_NAME}, in partial fulfillment of the requirements '
            "for the award of the degree, is a record of original work carried out by me under the guidance of "
            f"{GUIDE_NAME}. This work has not been submitted elsewhere for any other degree or diploma.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 1.3 * cm))
    story.append(Paragraph("Student Signature: ____________________    Date: ____________________", styles["BodyLeft"]))
    story.append(PageBreak())

    story.append(Paragraph("Certificate", styles["Heading1"]))
    story.append(
        Paragraph(
            f'This is to certify that the project report entitled "{PROJECT_TITLE}" has been completed by '
            f"{STUDENT_NAME} ({ROLL_NUMBER}) under my guidance and supervision. The work presented is found "
            "satisfactory and is approved for submission.",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 1.1 * cm))
    story.append(Paragraph(f"Guide Signature: ____________________    Name: {GUIDE_NAME}", styles["BodyLeft"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Head of Department Signature: ____________________", styles["BodyLeft"]))
    story.append(PageBreak())

    story.append(Paragraph("Acknowledgment", styles["Heading1"]))
    story.append(
        Paragraph(
            "I express my sincere gratitude to my project guide, faculty members, and department for their "
            "guidance and encouragement throughout this work. I also thank my friends and family for their "
            "constant support, and I acknowledge the open-source community whose tools and documentation helped "
            "in the successful development of this project.",
            styles["BodyJustify"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Abstract", styles["Heading1"]))
    story.append(
        Paragraph(
            "The AI-Powered Medical Appointment and Disease Risk Prediction System is a web-based healthcare "
            "application designed to streamline patient intake, appointment booking, and early disease risk "
            "assessment. The project addresses delays in manual triage by combining a trained Decision Tree "
            "model with rule-based guardrails so that high-risk cases can receive faster clinical attention. "
            "The implementation uses a hybrid FastAPI and Flask architecture: FastAPI exposes API and "
            "compatibility routes, while Flask handles the role-based portal, session management, dashboards, "
            "and patient-facing workflows. HTML/CSS/JavaScript templates provide the interface layer, CSV and "
            "SQLite files provide operational storage, and an optional OpenAI-backed patient chat can answer "
            "both contextual portal questions and general guidance prompts. Patients can register, submit "
            "health details, review risk summaries, chat with the assistant, book appointments, cancel "
            "appointments, and inspect booking history. Doctors and nurses receive dedicated dashboards to "
            "review cases, manage queue flow, update records, track leave dates, and apply emergency-availability "
            "rules. The trained model uses "
            f"{facts.feature_count} engineered features from a dataset of {facts.model_rows} usable records, "
            f"and the current evaluation reproduces a test accuracy of {facts.test_accuracy}%. The live "
            f"project snapshot currently includes {facts.profile_rows} patient profiles, {facts.appointment_rows} "
            f"appointments, {facts.doctor_profiles} doctor profiles, and {facts.template_count} rendered portal "
            "templates. The project demonstrates how AI can support faster decision-making in appointment "
            "management while remaining practical for small clinical environments. Future scope includes auth "
            "consolidation, stronger deployment security, notification services, richer explainability, and "
            "larger clinical datasets.",
            styles["BodyJustify"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    story.append(toc)
    story.append(PageBreak())

    story.append(Paragraph("Chapter 1: Introduction", styles["Heading1"]))
    story.append(Paragraph("1.1 Overview", styles["Heading2"]))
    story.append(
        Paragraph(
            "Healthcare systems frequently face delays when patients must manually describe symptoms and wait "
            "for administrative prioritization before a doctor reviews the case. This project proposes a "
            "single digital workflow where patient information, risk assessment, and appointment scheduling are "
            "combined into one coordinated platform.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("1.2 Problem Statement", styles["Heading2"]))
    story.append(
        Paragraph(
            "Traditional appointment booking and triage processes often depend on manual judgment, fragmented "
            "records, and repeated data entry. As a result, urgent patients may not be identified quickly, "
            "doctors may not get structured case summaries in time, and nurses may have limited support for "
            "queue management.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("1.3 Objectives", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Provide patient, doctor, and nurse portals in a single integrated healthcare workflow.",
                "Combine FastAPI API endpoints with Flask-rendered role-based portal pages.",
                "Predict disease risk level from patient symptoms and vital features using an ML model with safety guardrails.",
                "Automatically derive appointment priority, doctor availability decisions, and emergency escalation guidance.",
                "Store patient details, appointments, leave records, and review notes in a persistent data layer.",
                "Offer a practical base for future intelligent healthcare automation and deployment hardening.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("1.4 Scope", styles["Heading2"]))
    story.append(
        Paragraph(
            "The current implementation targets clinic-style workflow support. It covers patient registration, "
            "health-detail submission, disease risk prediction, appointment booking, doctor review, nurse "
            "queue handling, doctor leave management, emergency-doctor availability checks, patient booking "
            "history and cancellation, contextual patient chat, and compatibility API routes for prediction "
            "and booking operations.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("1.5 Limitations and Assumptions", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "The model is trained on a structured dataset rather than live hospital EHR data.",
                "The training snapshot is dominated by Low and Medium classes, although runtime guardrails can escalate cases to High Risk.",
                "Patient account handling currently spans both a portal registry CSV and a separate hashed API-auth CSV, and would benefit from consolidation.",
                "The application is suitable for academic or prototype use and would need stronger compliance controls for production healthcare deployment.",
                "Guide, student, and university details on the cover page are placeholders and can be updated easily.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Chapter 2: Literature Review / Existing System", styles["Heading1"]))
    story.append(Paragraph("2.1 Existing System", styles["Heading2"]))
    story.append(
        Paragraph(
            "Conventional appointment systems primarily focus on slot booking and usually leave triage as a "
            "manual task. Many small-scale systems also separate patient records, booking, and doctor review "
            "into different tools, which reduces efficiency and traceability.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("2.2 Limitations in Existing Approaches", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Urgent patients can be overlooked when prioritization is fully manual.",
                "Existing systems often lack integrated predictive analytics.",
                "Administrative staff may re-enter the same information in multiple places.",
                "Role-specific dashboards are often absent in basic booking platforms.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("2.3 Gap Analysis", styles["Heading2"]))
    story.append(
        Paragraph(
            "The proposed project fills the gap by linking health data capture, machine learning prediction, "
            "rule-based triage, and appointment allocation in one application. The result is a more structured "
            "and intelligent flow that benefits both patients and clinical staff.",
            styles["BodyJustify"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Chapter 3: System Analysis", styles["Heading1"]))
    story.append(Paragraph("3.1 Functional Requirements", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Requirement", "Description"],
                ["Patient registration and login", "Create accounts and access the patient workflow securely."],
                ["Health data entry", "Capture age, symptoms, glucose, BMI, blood pressure, sleep, and history."],
                ["Risk prediction", "Generate risk label, confidence breakdown, and guidance from the model."],
                ["Appointment booking and cancellation", "Book appointments, assign priority, and manage booking history."],
                ["Doctor dashboard", "Review appointments, risk details, and patient database information."],
                ["Nurse queue dashboard", "Inspect patient queue and update records before doctor review."],
                ["Doctor availability / leave", "Track leave dates and expose doctor availability status for scheduling decisions."],
                ["Emergency scheduling support", "Enforce emergency-doctor-only windows and recommend alternatives for high-risk cases."],
                ["AI chat support", "Provide contextual help for patient-side questions when configured."],
            ],
            [5.0 * cm, 11.3 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("3.2 Non-Functional Requirements", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Category", "Expectation"],
                ["Usability", "Simple role-based pages and guided health-detail forms."],
                ["Performance", "Fast prediction and scheduling response for individual patient requests."],
                ["Reliability", "Persistent storage using CSV files and SQLite tables with route-level validation."],
                ["Security", "Session guards, password policies, hashed doctor/nurse credentials, and environment-controlled secrets."],
                ["Maintainability", "Modular backend files for API entry, portal routes, prediction, triage, auth, and data access."],
            ],
            [4.4 * cm, 11.9 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("3.3 Feasibility Study", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Technical feasibility: achieved using Python, FastAPI, Flask, pandas, scikit-learn, and SQLite.",
                "Economic feasibility: suitable for academic implementation with low infrastructure cost.",
                "Operational feasibility: supports clinic-style users through separate patient, doctor, and nurse flows with lightweight file-backed storage.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("3.4 System Environment", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Item", "Details"],
                ["Programming Language", "Python"],
                ["Web Frameworks", "FastAPI + Flask"],
                ["Serving Layer", "Uvicorn with WSGIMiddleware for portal mounting"],
                ["Frontend", "HTML, CSS, JavaScript templates"],
                ["Data/Storage", "CSV files and SQLite"],
                ["ML Libraries", "scikit-learn, pandas, numpy, joblib"],
                ["Validation", "Pydantic"],
                ["Optional AI Service", "OpenAI-based patient chat when API key is configured"],
                ["Platform", "Web application"],
            ],
            [4.4 * cm, 11.9 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("3.5 Use Case / Flow Overview", styles["Heading2"]))
    story.append(build_flow_diagram())
    story.append(Paragraph("Figure 1: High-level process flow of the proposed system.", styles["Small"]))
    story.append(PageBreak())

    story.append(Paragraph("Chapter 4: System Design", styles["Heading1"]))
    story.append(Paragraph("4.1 Architectural Design", styles["Heading2"]))
    story.append(
        Paragraph(
            "The application follows a layered hybrid architecture. FastAPI acts as the API-facing entry layer, "
            "serves static assets, exposes prediction and booking endpoints, and mounts the Flask portal under "
            "a shared route surface. Flask coordinates server-rendered templates, role-based sessions, "
            "dashboard workflows, booking, cancellation, chat orchestration, and staff review actions. The "
            "prediction layer uses a Decision Tree pipeline with rule-based triage and emergency/availability "
            "logic. Persistent operational data is managed through CSV-based account and scheduling files plus "
            "an SQLite database for patient profiles and appointments.",
            styles["BodyJustify"],
        )
    )
    story.append(build_architecture_diagram())
    story.append(Paragraph("Figure 2: System architecture of the application.", styles["Small"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("4.2 Data Flow Design", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Patient data flows from form submission into feature normalization and persistence.",
                "Normalized features are passed to the prediction engine for risk computation.",
                "Risk level is translated into appointment priority by the triage module.",
                "Appointment requests are checked against doctor leave data and emergency-doctor scheduling rules.",
                "Appointments and profiles are stored for doctor and nurse dashboard access.",
                "Patient chat uses portal context first and optionally calls the OpenAI client for broader responses.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("4.3 Database Design", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Data Store", "Purpose"],
                ["patients.csv", "Portal patient registry, names, unique codes, and health-details completion flag"],
                ["patient_accounts.csv", "Separate hashed patient credential store used by the API auth manager"],
                ["doctor_accounts.csv", "Doctor authentication credentials"],
                ["nurse_accounts.csv", "Nurse authentication credentials"],
                ["doctor_profiles.csv", "Doctor metadata, specialization, available time, and emergency flag"],
                ["doctor_leave.csv", "Doctor leave dates used by scheduling checks"],
                ["new_patient_data.csv", "Latest submitted patient features plus nurse/doctor review fields"],
                ["patient_records.db / patient_profiles", "Structured health details and review history"],
                ["patient_records.db / appointments", "Booked appointments, priorities, and stored prediction output"],
            ],
            [6.0 * cm, 10.3 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("4.4 Current Stored Project Data Snapshot", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Artifact", "Current Count"],
                ["Portal patient registry rows", str(facts.portal_patient_accounts)],
                ["API patient auth rows", str(facts.api_patient_accounts)],
                ["Doctor accounts", str(facts.doctor_accounts)],
                ["Nurse accounts", str(facts.nurse_accounts)],
                ["Doctor profiles", str(facts.doctor_profiles)],
                ["Emergency doctor profiles", str(facts.emergency_doctors)],
                ["Doctor leave records", str(facts.doctor_leaves)],
                ["New patient health records", str(facts.new_patient_rows)],
                ["SQLite patient profiles", str(facts.profile_rows)],
                ["SQLite appointments", str(facts.appointment_rows)],
                ["HTML templates", str(facts.template_count)],
                ["Static assets", str(facts.static_asset_count)],
            ],
            [7.0 * cm, 6.0 * cm],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Chapter 5: Implementation", styles["Heading1"]))
    story.append(Paragraph("5.1 Tools and Technologies Used", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Layer", "Technology"],
                ["Frontend", "HTML templates, CSS, JavaScript"],
                ["API / Entry Layer", "FastAPI, Uvicorn, WSGIMiddleware"],
                ["Portal Layer", "Flask application with role-based routes and sessions"],
                ["Machine Learning", "Decision Tree pipeline with preprocessing and rule guardrails"],
                ["Storage", "CSV datasets and SQLite database"],
                ["AI Integration", "OpenAI Responses API for optional patient chat"],
                ["Utilities", "pandas, numpy, joblib, pydantic"],
            ],
            [4.2 * cm, 12.1 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("5.2 Modules / Components", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Module", "Purpose"],
                ["backend/main.py", "FastAPI entrypoint, API routes, static mount, portal mount, and compatibility redirects"],
                ["backend/flask_app.py", "Main portal workflows, routing, sessions, dashboards, booking, cancellation, and chat orchestration"],
                ["backend/predict.py", "Feature normalization, model prediction, and response shaping"],
                ["backend/triage.py", "Maps risk levels to appointment priority and slot recommendation"],
                ["backend/patient_db.py", "SQLite schema and patient/appointment persistence"],
                ["backend/patient_auth.py", "Separate patient auth manager with hashed credentials for API flows"],
                ["backend/doctor_auth.py", "Doctor signup and login with password hashing and validation"],
                ["backend/nurse_auth.py", "Nurse signup and login workflow"],
                ["backend/paths.py", "Environment-aware dataset paths and storage configuration"],
                ["backend/init_datasets.py", "Bootstrap script for required dataset CSV files"],
                ["frontend/templates", "Patient, doctor, and nurse interface pages"],
                ["frontend/static", "Shared assets, icons, and page animation files"],
            ],
            [5.5 * cm, 10.8 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("5.3 Implementation Highlights", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "FastAPI mounts the Flask portal and preserves legacy compatibility routes for earlier URLs.",
                "Portal routes support patient, doctor, and nurse role separation with session-based access guards.",
                "A trained model and rule guardrails work together to produce Low, Medium, or High risk output.",
                "Appointments include priority labels such as Immediate, Same Day, and Normal.",
                "Booking logic validates doctor leave data and enforces emergency-doctor-only night scheduling when required.",
                "Patient booking confirmation pages expose upcoming and past appointments, and allow appointment cancellation.",
                "Stored prediction data is accessible from dashboard workflows for follow-up actions.",
                "The patient portal can optionally use an OpenAI-backed chatbot with patient-context prompting.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("5.4 Security, Validation, and Data Handling", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Role-specific decorators protect patient, doctor, and nurse routes from cross-role access.",
                "Doctor and nurse credentials use salted PBKDF2 hashes, and a separate API patient auth manager also supports hashed patient credentials.",
                "Environment variables control dataset paths, Flask session security flags, and optional OpenAI settings.",
                "Patient data updates are synchronized to CSV storage and SQLite profile records for downstream dashboards.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Chapter 6: Testing", styles["Heading1"]))
    story.append(Paragraph("6.1 Testing Approach", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Functional testing of patient signup, login, and appointment workflow.",
                "Prediction testing using the trained Decision Tree model on held-out data.",
                "Dashboard testing for doctor and nurse visibility into queue and records.",
                "Scheduling-rule testing for leave handling and emergency-doctor availability checks.",
                "Persistence testing for CSV and SQLite updates after user actions.",
                "Patient chat testing for contextual replies and optional OpenAI fallback behavior.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("6.2 Sample Test Cases", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Test Case", "Expected Result", "Observed Outcome"],
                ["Patient submits health details", "Profile saved and prediction-ready features created", "Completed"],
                ["Risk prediction request", "Risk label, probability, and guidance returned", "Completed"],
                ["Appointment submission", "Appointment stored with priority and recommended slot", "Completed"],
                ["Night-time booking with non-emergency doctor", "System blocks booking and suggests emergency-capable alternatives", "Completed"],
                ["Doctor marked on leave", "System reports doctor unavailable and returns next-step guidance", "Completed"],
                ["Doctor dashboard access", "Doctor sees prioritized appointments and patient records", "Completed"],
                ["Nurse queue review", "Nurse can view and update patient intake details", "Completed"],
                ["Appointment cancellation", "Selected appointment is removed from patient history", "Completed"],
            ],
            [5.0 * cm, 6.0 * cm, 4.0 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("6.3 Model Evaluation", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Metric", "Value"],
                ["Usable records", str(facts.model_rows)],
                ["Training records", str(facts.train_rows)],
                ["Testing records", str(facts.test_rows)],
                ["Test accuracy", f"{facts.test_accuracy}%"],
                ["Precision", f"{facts.precision}%"],
                ["Recall", f"{facts.recall}%"],
                ["F1-score", f"{facts.f1_score}%"],
                ["Confusion matrix", facts.confusion_matrix_text],
            ],
            [5.2 * cm, 7.8 * cm],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Chapter 7: Results and Discussion", styles["Heading1"]))
    story.append(Paragraph("7.1 Dataset and Feature Summary", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Item", "Value"],
                ["Clean dataset shape", f"{facts.clean_rows} rows x {facts.clean_cols} columns"],
                ["Feature-engineered dataset shape", f"{facts.feature_rows} rows x {facts.feature_cols} columns"],
                ["Records used for training/evaluation", str(facts.model_rows)],
                ["Dropped rows due to missing values", str(facts.dropped_rows)],
                ["Model input feature count", str(facts.feature_count)],
                ["Risk distribution", ", ".join(f"{key}: {value}" for key, value in sorted(facts.risk_distribution.items()))],
            ],
            [6.0 * cm, 9.8 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("7.2 Output Discussion", styles["Heading2"]))
    story.append(
        Paragraph(
            "The system demonstrates strong predictive consistency on the available dataset and supports a "
            "workflow where model output is immediately usable for operational appointment decisions. The "
            "current dataset snapshot is dominated by Low and Medium risk patterns, and the resulting model "
            "performance reflects this distribution. Beyond classification, the project adds operational value "
            "through leave-aware scheduling, emergency-doctor recommendation, appointment history management, "
            "nurse review persistence, doctor dashboards, and patient-context chat support.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("7.3 Operational Coverage Snapshot", styles["Heading2"]))
    story.append(
        styled_table(
            [
                ["Operational Item", "Value"],
                ["Portal patient registry rows", str(facts.portal_patient_accounts)],
                ["API patient auth rows", str(facts.api_patient_accounts)],
                ["Doctor profiles / emergency doctors", f"{facts.doctor_profiles} / {facts.emergency_doctors}"],
                ["Doctor leave rows", str(facts.doctor_leaves)],
                ["Stored patient profiles / appointments", f"{facts.profile_rows} / {facts.appointment_rows}"],
                ["Portal templates / static assets", f"{facts.template_count} / {facts.static_asset_count}"],
            ],
            [6.5 * cm, 9.3 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("7.4 Performance Visualizations", styles["Heading2"]))
    for image_name, caption in [
        ("decision_tree_accuracy_loss.png", "Figure 3: Decision Tree accuracy versus loss."),
        ("decision_tree_metrics.png", "Figure 4: Decision Tree performance metrics."),
        ("decision_tree_confusion_matrix.png", "Figure 5: Confusion matrix of the trained model."),
    ]:
        image_path = BACKEND / image_name
        if image_path.exists():
            story.extend(figure_with_caption(image_path, caption, styles))
    story.append(PageBreak())

    story.append(Paragraph("Chapter 8: Conclusion and Future Work", styles["Heading1"]))
    story.append(Paragraph("8.1 Conclusion", styles["Heading2"]))
    story.append(
        Paragraph(
            "The AI-Powered Medical Appointment and Disease Risk Prediction System successfully integrates "
            "predictive analytics with a practical healthcare workflow. The project combines a hybrid FastAPI "
            "and Flask architecture, role-based access, risk assessment, triage, leave-aware appointment "
            "management, emergency scheduling logic, patient-context chat, and persistent storage in one working "
            "web application. The outcome is a strong academic prototype that demonstrates the value of "
            "AI-assisted decision support in healthcare operations.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("8.2 Future Work", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Use larger and clinically richer datasets to improve generalization.",
                "Expand the classifier to support more risk classes and disease categories.",
                "Add explainable AI views for doctor-facing decision support.",
                "Unify the separate patient-auth and portal-registry flows into one consistent authentication model.",
                "Integrate notification services for reminders and emergency escalation.",
                "Strengthen production readiness with audit logging and healthcare compliance measures.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("References / Bibliography", styles["Heading1"]))
    story.extend(
        bullet_points(
            [
                "FastAPI Documentation.",
                "Flask Documentation.",
                "pandas Documentation.",
                "scikit-learn Documentation.",
                "SQLite Documentation.",
                "Uvicorn and ASGI integration references.",
                "OpenAI API platform documentation for optional chatbot integration.",
                "Project source code, templates, datasets, and trained artifacts in the current workspace.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Appendices", styles["Heading1"]))
    story.append(Paragraph("Appendix A: Installation Guide", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Create and activate a Python virtual environment.",
                "Install project dependencies used by the FastAPI entry layer, Flask portal, and ML pipeline.",
                "Initialize dataset CSV files using the dataset bootstrap script when needed.",
                "Configure environment variables such as CSV paths and optional OpenAI settings.",
                "Run the project through backend/main.py so both API and portal routes are available.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("Appendix B: Major Routes and Features", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "FastAPI entry routes cover predict-risk, book-appointment, patient-features lookup, appointments listing, static serving, and portal mounting.",
                "Patient routes cover signup, login, health details, AI chat, booking, booking confirmation, cancellation, feature updates, and emergency-doctor discovery.",
                "Doctor routes cover signup, login, dashboard, appointments, patient database, leave management, availability status, and risk review.",
                "Nurse routes cover signup, login, dashboard, appointment queue, patient-record updates, and risk review.",
                "Compatibility routes preserve earlier URLs by redirecting them into the shared portal flow.",
            ],
            styles["BodyLeft"],
        )
    )
    story.append(Paragraph("Appendix C: Project Artifacts", styles["Heading2"]))
    story.extend(
        bullet_points(
            [
                "Training and preprocessing artifacts include Healthcare.csv, Healthcare_Cleaned.csv, Healthcare_FeatureEngineered.csv, the trained model, and the label encoder.",
                "Evaluation artifacts include the accuracy/loss chart, metrics summary, and confusion matrix images bundled in the backend directory.",
                "Operational datasets under backend/datasets store patient registry rows, auth files, doctor profiles, leave records, health submissions, and SQLite appointment data.",
            ],
            styles["BodyLeft"],
        )
    )
    return story


def generate_pdf():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    facts = gather_project_facts()
    doc = NumberedCanvasDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2.4 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title=PROJECT_TITLE,
        author=STUDENT_NAME,
    )
    story = build_story(facts)
    doc.multiBuild(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    pdf_path = generate_pdf()
    print(f"Generated report: {pdf_path}")
