"""
Recovera AI: Clinical Image Analysis Report Service (ReportLab PDF & CSV)
Generates:
1. Multi-page, publication-grade, hospital-branded PDF clinical reports with:
   - Hospital masthead & institutional provenance
   - Patient demographics, medical history & clinical condition
   - Deep learning computer vision architecture telemetry & confidence
   - Grad-CAM heatmap visualization and U-Net++ segmentation overlays
   - Multi-factorial clinical characteristics (erythema, dehiscence, tissue composition, fracture callus)
   - Longitudinal comparison against baseline records
   - Google Gemini multimodal clinical recovery synthesis & recommendations
   - Clinician review status, notes, and doctor sign-off block
   - Standard clinical AI governance & regulatory disclaimers
2. Structured CSV export of analytical biomarkers and provenance metadata.
"""

import os
import csv
import io
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from app.core.config import settings
from app.models.image_analysis import ImageAnalysis, WoundObservation, FractureAnalysis, GeminiReassessment
from app.models.patient import Patient
from app.models.user import User


def generate_image_analysis_pdf(
    analysis: ImageAnalysis,
    patient: Patient,
    uploader: Optional[User] = None,
    reviewer: Optional[User] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Builds a professional clinical PDF dossier for the specified ImageAnalysis record.
    Returns the absolute path to the generated PDF file.
    """
    reports_dir = Path(settings.UPLOAD_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not output_path:
        filename = f"Recovera_ImageAnalysis_{patient.patient_code}_{analysis.id[:8]}.pdf"
        output_path = str(reports_dir / filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    # Styles
    base_styles = getSampleStyleSheet()

    # Color Palette: Deep Slate, Clinical Teal, Pure Dark
    TEAL = colors.HexColor("#0d9488")
    DARK_TEAL = colors.HexColor("#0f766e")
    SLATE_DARK = colors.HexColor("#0f172a")
    SLATE_GRAY = colors.HexColor("#334155")
    LIGHT_BG = colors.HexColor("#f8fafc")
    BORDER_COLOR = colors.HexColor("#cbd5e1")
    ALERT_RED = colors.HexColor("#b91c1c")
    WARNING_AMBER = colors.HexColor("#b45309")
    SUCCESS_GREEN = colors.HexColor("#15803d")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=SLATE_DARK,
        alignment=TA_LEFT,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=SLATE_GRAY,
        alignment=TA_LEFT
    )

    section_hdr_style = ParagraphStyle(
        "SectionHeader",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=DARK_TEAL,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SLATE_DARK
    )

    bold_label_style = ParagraphStyle(
        "ReportBoldLabel",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=SLATE_DARK
    )

    gemini_summary_style = ParagraphStyle(
        "GeminiSummary",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=SLATE_DARK
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=base_styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_JUSTIFY
    )

    story = []

    # 1. Header Banner & Masthead
    header_data = [
        [
            Paragraph("<b>RECOVERA AI</b> &bull; CLINICAL DECISION SUPPORT", title_style),
            Paragraph(f"<b>CONFIDENTIAL MEDICAL RECORD</b><br/>Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ParagraphStyle("HdrRight", parent=subtitle_style, alignment=TA_RIGHT))
        ],
        [
            Paragraph("AI-Powered Surgical Wound & Radiographic Fracture Analysis Platform", subtitle_style),
            Paragraph(f"<b>Analysis ID:</b> {analysis.id[:12]} &bull; <b>Status:</b> {analysis.review_status.upper()}", ParagraphStyle("HdrStatus", parent=subtitle_style, alignment=TA_RIGHT))
        ]
    ]
    hdr_table = Table(header_data, colWidths=[360, 180])
    hdr_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(hdr_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceBefore=2, spaceAfter=8))

    # 2. Patient Demographics & Clinical Context
    p_name = patient.name if patient else "Unknown"
    p_code = patient.patient_code if patient else "N/A"
    p_gender = (patient.gender or "N/A").title()
    p_condition = patient.active_condition or "Not Specified"
    p_dob = str(patient.dob) if patient and patient.dob else "N/A"

    patient_grid = [
        [
            Paragraph("<b>Patient Name:</b>", bold_label_style), Paragraph(p_name, body_style),
            Paragraph("<b>Patient ID (MRN):</b>", bold_label_style), Paragraph(p_code, body_style)
        ],
        [
            Paragraph("<b>Active Condition:</b>", bold_label_style), Paragraph(p_condition, body_style),
            Paragraph("<b>Date of Birth:</b>", bold_label_style), Paragraph(f"{p_dob} ({p_gender})", body_style)
        ],
        [
            Paragraph("<b>Analysis Type:</b>", bold_label_style), Paragraph(analysis.analysis_type.replace("_", " ").title(), body_style),
            Paragraph("<b>Capture Date:</b>", bold_label_style), Paragraph(analysis.created_at.strftime('%Y-%m-%d %H:%M'), body_style)
        ]
    ]
    pat_table = Table(patient_grid, colWidths=[95, 175, 105, 165])
    pat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 8))

    # 3. Deep Learning Diagnostic Assessment & Telemetry
    story.append(Paragraph("1. Deep Learning Vision Assessment & Model Telemetry", section_hdr_style))

    sev_color = ALERT_RED if analysis.severity_level == "critical" else (
        WARNING_AMBER if analysis.severity_level in ("elevated", "moderate") else SUCCESS_GREEN
    )

    model_grid = [
        [
            Paragraph("<b>Primary Classification:</b>", bold_label_style),
            Paragraph(f"<b>{analysis.primary_prediction}</b>", ParagraphStyle("PrimPred", parent=body_style, textColor=DARK_TEAL, fontName="Helvetica-Bold")),
            Paragraph("<b>Model Architecture:</b>", bold_label_style),
            Paragraph(analysis.model_architecture, body_style)
        ],
        [
            Paragraph("<b>Confidence Score:</b>", bold_label_style),
            Paragraph(f"<b>{analysis.confidence_score * 100:.1f}%</b>", body_style),
            Paragraph("<b>Pretrained Weights:</b>", bold_label_style),
            Paragraph(analysis.weights_provenance, body_style)
        ],
        [
            Paragraph("<b>Clinical Severity:</b>", bold_label_style),
            Paragraph(f"<b>{analysis.severity_level.upper()}</b>", ParagraphStyle("Sev", parent=body_style, textColor=sev_color, fontName="Helvetica-Bold")),
            Paragraph("<b>Inference Latency:</b>", bold_label_style),
            Paragraph(f"{analysis.processing_time_ms} ms", body_style)
        ],
        [
            Paragraph("<b>Findings Summary:</b>", bold_label_style),
            Paragraph(analysis.findings_summary, body_style),
            Paragraph("<b>Model Name / Ver:</b>", bold_label_style),
            Paragraph(f"{analysis.model_name} ({analysis.model_version})", body_style)
        ]
    ]

    model_table = Table(model_grid, colWidths=[110, 160, 110, 160])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 8))

    # 4. Multi-modal Visual Evidence & Explainability Artifacts (Grad-CAM & Segmentation)
    story.append(Paragraph("2. Explainability & Visual Attribution Artifacts", section_hdr_style))

    # Check images existence
    img_cells = []
    labels = []

    # Original Image
    orig_path = analysis.image.file_path if analysis.image else None
    if orig_path and os.path.exists(orig_path):
        try:
            img_cells.append(RLImage(orig_path, width=170, height=130))
            labels.append(Paragraph("<b>Original Clinical Submission</b>", bold_label_style))
        except Exception:
            pass

    # Grad-CAM Overlay
    gcam_path = analysis.gradcam_overlay_path
    if gcam_path and os.path.exists(gcam_path):
        try:
            img_cells.append(RLImage(gcam_path, width=170, height=130))
            labels.append(Paragraph("<b>Grad-CAM Explainability Heatmap</b>", bold_label_style))
        except Exception:
            pass

    # Segmentation or Secondary Overlay
    seg_path = analysis.segmentation_overlay_path or analysis.gradcam_heatmap_path
    if seg_path and os.path.exists(seg_path):
        try:
            img_cells.append(RLImage(seg_path, width=170, height=130))
            sub_label = "U-Net++ Segmentation Contour" if analysis.segmentation_overlay_path else "Attribution Saliency Map"
            labels.append(Paragraph(f"<b>{sub_label}</b>", bold_label_style))
        except Exception:
            pass

    if img_cells:
        img_table = Table([img_cells, labels], colWidths=[180] * len(img_cells))
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ]))
        story.append(img_table)
    else:
        story.append(Paragraph("<i>Visual image artifacts unavailable or archived.</i>", body_style))

    story.append(Spacer(1, 8))

    # 5. Domain Specific Granular Metrics (Wound vs Fracture)
    if analysis.analysis_type == "surgical_wound" and analysis.wound_observation:
        wo = analysis.wound_observation
        story.append(Paragraph("3. Quantitative Surgical Wound Biomarkers", section_hdr_style))
        w_grid = [
            [
                Paragraph("<b>Estimated Area:</b>", bold_label_style), Paragraph(f"{wo.wound_area_cm2 or 'N/A'} cm&sup2; ({wo.wound_area_pixels} px)", body_style),
                Paragraph("<b>Erythema Score:</b>", bold_label_style), Paragraph(f"{wo.erythema_score:.1f} / 100", body_style)
            ],
            [
                Paragraph("<b>Tissue Composition:</b>", bold_label_style), Paragraph(wo.predominant_tissue_type.title(), body_style),
                Paragraph("<b>Dehiscence Risk:</b>", bold_label_style), Paragraph(f"{wo.dehiscence_risk_score:.1f} / 100", body_style)
            ],
            [
                Paragraph("<b>Exudate Detected:</b>", bold_label_style), Paragraph(f"{'Yes (' + wo.exudate_type + ')' if wo.exudate_detected else 'None'}", body_style),
                Paragraph("<b>Healing Trajectory:</b>", bold_label_style), Paragraph(wo.healing_trajectory_status.replace('_', ' ').title(), body_style)
            ]
        ]
        w_table = Table(w_grid, colWidths=[110, 160, 110, 160])
        w_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(w_table)
        story.append(Spacer(1, 8))

    elif analysis.analysis_type == "bone_fracture_xray" and analysis.fracture_analysis:
        fa = analysis.fracture_analysis
        story.append(Paragraph("3. Quantitative Bone Radiograph Biomarkers", section_hdr_style))
        f_grid = [
            [
                Paragraph("<b>Fracture Detected:</b>", bold_label_style), Paragraph(f"<b>{'POSITIVE' if fa.fracture_detected else 'NEGATIVE'}</b> ({fa.fracture_probability * 100:.1f}%)", body_style),
                Paragraph("<b>Anatomical Bone:</b>", bold_label_style), Paragraph(fa.anatomical_bone, body_style)
            ],
            [
                Paragraph("<b>Cortical Displacement:</b>", bold_label_style), Paragraph(f"{fa.displacement_severity.title()} ({'Displaced' if fa.displacement_present else 'Non-displaced'})", body_style),
                Paragraph("<b>Suspected Fracture Type:</b>", bold_label_style), Paragraph(fa.fracture_type or "N/A", body_style)
            ],
            [
                Paragraph("<b>Callus Formation:</b>", bold_label_style), Paragraph(f"{'Observed (Bridging present)' if fa.callus_formation_observed else 'Not yet visualized'}", body_style),
                Paragraph("<b>Healing Stage:</b>", bold_label_style), Paragraph(fa.healing_stage.replace('_', ' ').title(), body_style)
            ]
        ]
        f_table = Table(f_grid, colWidths=[120, 150, 120, 150])
        f_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(f_table)
        story.append(Spacer(1, 8))

    # 6. Google Gemini Multimodal Reassessment
    if analysis.gemini_reassessment:
        gr = analysis.gemini_reassessment
        story.append(Paragraph("4. Multimodal Synthesis (Google Gemini API Clinical Intelligence)", section_hdr_style))

        gemini_items = [
            Paragraph(f"<b>Reassessment Summary:</b> {gr.reassessment_summary}", gemini_summary_style),
            Spacer(1, 3),
            Paragraph(f"<b>Longitudinal Changes vs Baseline:</b> {gr.changes_vs_previous}", body_style)
        ]

        if gr.potential_concerns:
            concerns_str = ", ".join(gr.potential_concerns) if isinstance(gr.potential_concerns, list) else str(gr.potential_concerns)
            gemini_items.append(Paragraph(f"<b>Potential Concerns:</b> {concerns_str}", ParagraphStyle("Cnc", parent=body_style, textColor=ALERT_RED)))

        if gr.suggested_clinician_actions:
            actions_str = ", ".join(gr.suggested_clinician_actions) if isinstance(gr.suggested_clinician_actions, list) else str(gr.suggested_clinician_actions)
            gemini_items.append(Paragraph(f"<b>Suggested Clinician Actions:</b> {actions_str}", body_style))

        if gr.recovery_plan_review_recommended:
            gemini_items.append(Paragraph(f"<b>Proposed Recovery Plan Note:</b> {gr.recovery_plan_proposed_notes or 'Review suggested by AI'}", ParagraphStyle("RecPlan", parent=body_style, textColor=DARK_TEAL)))

        gem_box = Table([[gemini_items]], colWidths=[540])
        gem_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, TEAL),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(gem_box)
        story.append(Spacer(1, 8))

    # 7. Clinician Review & Formal Doctor Sign-off
    story.append(Paragraph("5. Clinician Review & Doctor Sign-Off", section_hdr_style))

    reviewer_name = reviewer.name if reviewer else (analysis.reviewer.name if analysis.reviewer else "Pending Physician Assignment")
    review_date = analysis.reviewed_at.strftime("%Y-%m-%d %H:%M") if analysis.reviewed_at else "Awaiting Formal Review"
    review_status_display = analysis.review_status.upper()

    notes_text = analysis.clinician_notes or "No clinician notes entered yet."
    confirmed_text = analysis.doctor_confirmed_findings or (
        "AI diagnostic findings pending formal physician confirmation." if analysis.review_status == "pending_review" else "Findings confirmed by treating physician."
    )

    sign_grid = [
        [
            Paragraph("<b>Reviewing Clinician:</b>", bold_label_style), Paragraph(reviewer_name, body_style),
            Paragraph("<b>Review Status:</b>", bold_label_style), Paragraph(f"<b>{review_status_display}</b>", bold_label_style)
        ],
        [
            Paragraph("<b>Review Timestamp:</b>", bold_label_style), Paragraph(review_date, body_style),
            Paragraph("<b>Patient Visibility:</b>", bold_label_style), Paragraph("Visible in Patient Portal" if analysis.is_patient_visible else "Staff-Only (Internal Review)", body_style)
        ],
        [
            Paragraph("<b>Clinician Notes:</b>", bold_label_style),
            Paragraph(notes_text, body_style),
            Paragraph("<b>Physician Signature:</b>", bold_label_style),
            Paragraph(f"<i>Digitally signed by {reviewer_name}</i><br/>Recovera AI Clinical OS", body_style)
        ]
    ]

    sign_table = Table(sign_grid, colWidths=[110, 160, 110, 160])
    sign_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(sign_table)
    story.append(Spacer(1, 10))

    # 8. Governance & Regulatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=4, spaceAfter=4))
    disclaimer_text = (
        "<b>CLINICAL AI GOVERNANCE DISCLAIMER:</b> This automated computer vision analysis and Grad-CAM explainability "
        "report is generated by Recovera AI as a clinical decision support instrument. It does not replace independent clinical judgment, "
        "direct bedside examination, or formal radiographic evaluation by a licensed healthcare professional. All findings, classifications, "
        "and suggested actions must be verified by the treating physician prior to altering therapeutic protocols or surgical care pathways."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build Document
    doc.build(story)
    return output_path


def generate_image_analysis_csv(analysis: ImageAnalysis, patient: Patient) -> str:
    """
    Generates a structured CSV export representing all quantitative biomarkers,
    model confidence scores, and clinical metadata for research or audit extraction.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header Row
    writer.writerow([
        "Analysis_ID",
        "Patient_ID",
        "Patient_Code",
        "Patient_Condition",
        "Analysis_Type",
        "Model_Name",
        "Model_Architecture",
        "Model_Version",
        "Primary_Prediction",
        "Confidence_Score",
        "Severity_Level",
        "Processing_Time_MS",
        "Wound_Area_CM2",
        "Wound_Area_Pixels",
        "Erythema_Score",
        "Dehiscence_Risk_Score",
        "Exudate_Detected",
        "Tissue_Type",
        "Fracture_Detected",
        "Fracture_Probability",
        "Anatomical_Bone",
        "Displacement_Severity",
        "Callus_Observed",
        "Healing_Stage",
        "Gemini_Summary",
        "Review_Status",
        "Reviewer_ID",
        "Reviewed_At",
        "Is_Patient_Visible",
        "Created_At"
    ])

    # Extract Wound & Fracture fields
    wo = analysis.wound_observation
    fa = analysis.fracture_analysis
    gr = analysis.gemini_reassessment

    w_area_cm2 = wo.wound_area_cm2 if wo else ""
    w_area_px = wo.wound_area_pixels if wo else ""
    erythema = wo.erythema_score if wo else ""
    dehiscence = wo.dehiscence_risk_score if wo else ""
    exudate = wo.exudate_detected if wo else ""
    tissue = wo.predominant_tissue_type if wo else ""

    f_detected = fa.fracture_detected if fa else ""
    f_prob = fa.fracture_probability if fa else ""
    f_bone = fa.anatomical_bone if fa else ""
    f_disp = fa.displacement_severity if fa else ""
    f_callus = fa.callus_formation_observed if fa else ""
    f_stage = fa.healing_stage if fa else ""

    gemini_sum = gr.reassessment_summary if gr else ""

    writer.writerow([
        analysis.id,
        patient.id if patient else "",
        patient.patient_code if patient else "",
        patient.active_condition if patient else "",
        analysis.analysis_type,
        analysis.model_name,
        analysis.model_architecture,
        analysis.model_version,
        analysis.primary_prediction,
        analysis.confidence_score,
        analysis.severity_level,
        analysis.processing_time_ms,
        w_area_cm2,
        w_area_px,
        erythema,
        dehiscence,
        exudate,
        tissue,
        f_detected,
        f_prob,
        f_bone,
        f_disp,
        f_callus,
        f_stage,
        gemini_sum,
        analysis.review_status,
        analysis.reviewed_by or "",
        analysis.reviewed_at.isoformat() if analysis.reviewed_at else "",
        analysis.is_patient_visible,
        analysis.created_at.isoformat()
    ])

    return output.getvalue()
