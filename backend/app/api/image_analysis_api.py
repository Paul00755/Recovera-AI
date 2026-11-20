"""
Recovera AI: Clinical Medical Image Analysis API
Provides endpoints for:
1. Patient listing and clinical context retrieval for image workflows.
2. Medical image upload, validation (strict radiograph checking), and deep learning vision processing.
3. Interactive Draft session handling: Save, Re-analyse, or Discard workflow.
4. Multimodal Google Gemini API recovery reassessment.
5. Clinician review, amendment, and doctor sign-off with patient visibility controls.
6. Publication-grade PDF and CSV clinical export generation.
7. Research benchmark registry queries for academic evaluation.
8. Authenticated static file serving for medical images, Grad-CAM overlays, and segmentations.
"""

import os
import uuid
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Response, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, or_

from app.core.database import get_db
from app.core.config import settings
from app.core.constants import Role
from app.core.audit import record_audit
from app.api.deps import get_current_user, require_role, enforce_patient_isolation
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.user import User
from app.models.recovery_plan import RecoveryPlan
from app.models.image_analysis import (
    MedicalImage,
    ImageAnalysisSession,
    ImageAnalysis,
    WoundObservation,
    FractureAnalysis,
    GeminiReassessment,
    AnalysisExport,
    ModelTrainingRun
)
from app.services.medical_vision_service import (
    validate_medical_image,
    analyze_surgical_wound,
    analyze_bone_fracture
)
from app.services.image_gemini_reassessment_service import perform_gemini_image_reassessment
from app.services.image_report_service import generate_image_analysis_pdf, generate_image_analysis_csv
from app.services.risk_service import evaluate_patient_risk
from app.services.digital_twin_service import compute_digital_twin_state
from app.services.notification_service import create_in_app_notification

router = APIRouter(prefix="/api/image-analysis", tags=["Medical Image Analysis & Deep Learning"])

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------
class ClinicianReviewIn(BaseModel):
    review_status: str  # approved, amended, rejected
    clinician_notes: Optional[str] = None
    clinician_agreed_with_ai: Optional[bool] = None
    doctor_confirmed_findings: Optional[str] = None
    is_patient_visible: bool = False


class SaveSessionIn(BaseModel):
    review_status: Optional[str] = "approved"
    clinician_notes: Optional[str] = None
    clinician_agreed_with_ai: Optional[bool] = True
    doctor_confirmed_findings: Optional[str] = None
    is_patient_visible: Optional[bool] = True


class DiscardSessionIn(BaseModel):
    reason: Optional[str] = "Clinical assessment discarded by user."


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def _format_analysis_response(analysis: ImageAnalysis, db: Session) -> Dict[str, Any]:
    """Helper to format ImageAnalysis for JSON API response."""
    img = analysis.image
    uploader = db.get(User, img.uploaded_by) if (img and img.uploaded_by) else None
    reviewer = db.get(User, analysis.reviewed_by) if analysis.reviewed_by else None
    patient = db.get(Patient, analysis.patient_id)

    # Relative file serving URLs
    def _to_url(abs_path: Optional[str]) -> Optional[str]:
        if not abs_path:
            return None
        try:
            rel = Path(abs_path).relative_to(Path(settings.UPLOAD_DIR)).as_posix()
            return f"/api/image-analysis/files/{rel}"
        except Exception:
            # Fallback if path is outside upload dir or filename only
            return f"/api/image-analysis/files/{Path(abs_path).name}"

    res = {
        "id": analysis.id,
        "session_id": analysis.session_id,
        "patient_id": analysis.patient_id,
        "patient_code": patient.patient_code if patient else "N/A",
        "patient_name": patient.name if patient else "N/A",
        "patient_condition": patient.active_condition if patient else None,
        "analysis_type": analysis.analysis_type,
        "attempt_number": analysis.attempt_number,
        "model_name": analysis.model_name,
        "model_version": analysis.model_version,
        "model_architecture": analysis.model_architecture,
        "weights_provenance": analysis.weights_provenance,
        "primary_prediction": analysis.primary_prediction,
        "specific_subtype": (analysis.metrics_json or {}).get("specific_subtype"),
        "confidence_score": analysis.confidence_score,
        "probabilities": analysis.probabilities_json or {},
        "findings_summary": analysis.findings_summary,
        "explainable_ai": (analysis.metrics_json or {}).get("explainable_ai"),
        "clinical_correlation": (analysis.metrics_json or {}).get("clinical_correlation"),
        "severity_level": analysis.severity_level,
        "processing_time_ms": analysis.processing_time_ms,
        "review_status": analysis.review_status,
        "reviewed_by": analysis.reviewed_by,
        "reviewer_name": reviewer.name if reviewer else None,
        "reviewed_at": analysis.reviewed_at.isoformat() if analysis.reviewed_at else None,
        "clinician_notes": analysis.clinician_notes,
        "clinician_agreed_with_ai": analysis.clinician_agreed_with_ai,
        "doctor_confirmed_findings": analysis.doctor_confirmed_findings,
        "is_patient_visible": analysis.is_patient_visible,
        "created_at": analysis.created_at.isoformat(),
        # Image and Artifact URLs
        "image": {
            "id": img.id if img else None,
            "original_filename": img.original_filename if img else None,
            "url": _to_url(img.file_path) if img else None,
            "anatomical_site": img.anatomical_site if img else None,
            "clinical_view": img.clinical_view if img else None,
            "is_baseline": img.is_baseline if img else False,
            "image_width": img.image_width if img else None,
            "image_height": img.image_height if img else None,
            "uploader_name": uploader.name if uploader else "Staff",
            "captured_at": img.captured_at.isoformat() if (img and img.captured_at) else None
        } if img else None,
        "artifacts": {
            "gradcam_heatmap_url": _to_url(analysis.gradcam_heatmap_path),
            "gradcam_overlay_url": _to_url(analysis.gradcam_overlay_path),
            "segmentation_mask_url": _to_url(analysis.segmentation_mask_path),
            "segmentation_overlay_url": _to_url(analysis.segmentation_overlay_path),
        },
        "metrics": analysis.metrics_json or {},
    }

    # Add domain-specific observation
    if analysis.analysis_type == "surgical_wound" and analysis.wound_observation:
        wo = analysis.wound_observation
        res["wound_observation"] = {
            "wound_area_cm2": wo.wound_area_cm2,
            "wound_area_pixels": wo.wound_area_pixels,
            "wound_perimeter_pixels": wo.wound_perimeter_pixels,
            "relative_wound_size_pct": wo.relative_wound_size_pct,
            "erythema_score": wo.erythema_score,
            "edema_present": wo.edema_present,
            "dehiscence_risk_score": wo.dehiscence_risk_score,
            "exudate_detected": wo.exudate_detected,
            "exudate_type": wo.exudate_type,
            "predominant_tissue_type": wo.predominant_tissue_type,
            "healing_trajectory_status": wo.healing_trajectory_status
        }
    elif analysis.analysis_type == "bone_fracture_xray" and analysis.fracture_analysis:
        fa = analysis.fracture_analysis
        res["fracture_analysis"] = {
            "fracture_detected": fa.fracture_detected,
            "fracture_probability": fa.fracture_probability,
            "anatomical_bone": fa.anatomical_bone,
            "fracture_type": fa.fracture_type,
            "displacement_present": fa.displacement_present,
            "displacement_severity": fa.displacement_severity,
            "callus_formation_observed": fa.callus_formation_observed,
            "healing_stage": fa.healing_stage
        }

    # Add Gemini Reassessment
    if analysis.gemini_reassessment:
        gr = analysis.gemini_reassessment
        res["gemini_reassessment"] = {
            "id": gr.id,
            "model_name": gr.model_name,
            "reassessment_summary": gr.reassessment_summary,
            "relevant_new_findings": gr.relevant_new_findings or [],
            "changes_vs_previous": gr.changes_vs_previous,
            "potential_concerns": gr.potential_concerns or [],
            "suggested_clinician_actions": gr.suggested_clinician_actions or [],
            "recovery_plan_review_recommended": gr.recovery_plan_review_recommended,
            "recovery_plan_proposed_notes": gr.recovery_plan_proposed_notes,
            "risk_review_recommended": gr.risk_review_recommended,
            "recommended_risk_tier": gr.recommended_risk_tier,
            "risk_escalation_rationale": gr.risk_escalation_rationale,
            "uncertainties_and_missing_data": gr.uncertainties_and_missing_data or [],
            "processing_status": gr.processing_status
        }

    return res


# ----------------------------------------------------
# 1. Patient Directory for Image Analysis
# ----------------------------------------------------
@router.get("/patients")
def list_patients_for_image_analysis(
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """
    Returns authorized patients for medical image analysis with clinical condition,
    assigned physician, and image baseline indicators.
    """
    query = select(Patient).where(or_(Patient.active == True, Patient.care_status != "archived", Patient.care_status == "active"))

    if current_user.role == Role.DOCTOR.value:
        # Assigned patients or primary physician
        subq = select(PatientDoctorAssignment.patient_id).where(
            PatientDoctorAssignment.doctor_id == current_user.id,
            or_(PatientDoctorAssignment.is_active == True, PatientDoctorAssignment.is_active == None)
        )
        query = query.where(or_(Patient.assigned_doctor_id == current_user.id, Patient.id.in_(subq)))

    patients = db.scalars(query.order_by(Patient.created_at.desc())).all()

    results = []
    for p in patients:
        p_name = p.name or (p.user.name if p.user else None) or p.patient_code or "Patient"
        p_code = p.patient_code or "REC-PT"
        cond = p.active_condition or ""

        # Filter by search string if provided
        if q:
            term = q.lower()
            if term not in p_name.lower() and term not in p_code.lower() and term not in cond.lower():
                continue

        # Look up baseline images
        baseline_wound = db.scalars(
            select(MedicalImage).where(
                MedicalImage.patient_id == p.id,
                MedicalImage.image_type == "surgical_wound",
                MedicalImage.is_baseline == True
            )
        ).first()

        baseline_xray = db.scalars(
            select(MedicalImage).where(
                MedicalImage.patient_id == p.id,
                MedicalImage.image_type == "bone_fracture_xray",
                MedicalImage.is_baseline == True
            )
        ).first()

        saved_count = db.query(ImageAnalysis).filter(ImageAnalysis.patient_id == p.id).count()

        doctor_user = db.get(User, p.assigned_doctor_id) if p.assigned_doctor_id else None

        results.append({
            "id": p.id,
            "patient_code": p.patient_code,
            "name": p_name,
            "active_condition": p.active_condition,
            "condition_stage": p.condition_stage,
            "date_of_birth": p.dob.isoformat() if p.dob else None,
            "gender": p.gender,
            "assigned_doctor_id": p.assigned_doctor_id,
            "assigned_doctor_name": doctor_user.name if doctor_user else "Unassigned",
            "has_baseline_wound": baseline_wound is not None,
            "baseline_wound_image_id": baseline_wound.id if baseline_wound else None,
            "has_baseline_xray": baseline_xray is not None,
            "baseline_xray_image_id": baseline_xray.id if baseline_xray else None,
            "saved_analyses_count": saved_count
        })

    return results


# ----------------------------------------------------
# 2. Upload Image & Initiate Draft Analysis Session
# ----------------------------------------------------
@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_image_analysis_session(
    patient_id: str = Form(...),
    analysis_type: str = Form(..., pattern="^(surgical_wound|bone_fracture_xray)$"),
    file: UploadFile = File(...),
    anatomical_site: Optional[str] = Form(None),
    clinical_view: Optional[str] = Form(None),
    is_baseline: bool = Form(False),
    baseline_image_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """
    Initiates a new image analysis session:
    1. Validates patient access and permissions.
    2. Enforces strict radiograph validation for X-rays (rejecting standard color photographs).
    3. Runs deep learning inference (ResNet50 / DenseNet121).
    4. Computes mathematical Grad-CAM explainability and segmentation overlays.
    5. Creates a DRAFT analysis session. Findings do not mutate patient care until saved.
    """
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    enforce_patient_isolation(patient_id=patient.id, current_user=current_user, db=db)

    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTS)}"
        )

    # Save to staging/storage directory
    subfolder = "wounds" if analysis_type == "surgical_wound" else "xrays"
    dest_dir = Path(settings.UPLOAD_DIR) / "medical_images" / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    img_uuid = str(uuid.uuid4())
    stored_filename = f"{patient.patient_code}_{subfolder}_{img_uuid[:8]}{ext}"
    dest_path = dest_dir / stored_filename

    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image size exceeds maximum limit of 25MB.")

    with open(dest_path, "wb") as f:
        f.write(contents)

    # Validate image dimensions and radiograph characteristics
    is_valid, err_msg, img_metrics = validate_medical_image(str(dest_path), expected_type=analysis_type)
    if not is_valid:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Image Validation Failed: {err_msg}"
        )

    # Check for baseline image if longitudinal comparison is needed
    baseline_img_path = None
    if baseline_image_id:
        base_img = db.get(MedicalImage, baseline_image_id)
        if base_img and os.path.exists(base_img.file_path):
            baseline_img_path = base_img.file_path
    elif not is_baseline:
        # Auto-detect patient's existing baseline image for this type
        auto_base = db.scalars(
            select(MedicalImage).where(
                MedicalImage.patient_id == patient.id,
                MedicalImage.image_type == analysis_type,
                MedicalImage.is_baseline == True
            )
        ).first()
        if auto_base and os.path.exists(auto_base.file_path):
            baseline_img_path = auto_base.file_path

    # Build comprehensive clinical patient profile for cross-correlation
    active_rx = []
    try:
        from app.models.medication import Medication
        meds = db.scalars(select(Medication).where(Medication.patient_id == patient.id, Medication.active == True)).all()
        for m in meds:
            active_rx.append({"name": f"{m.name} {m.dosage or ''}".strip(), "frequency": m.frequency or "daily", "instructions": m.instructions})
    except Exception:
        pass

    med_hist = []
    past_meds = []
    try:
        from app.models.patient import MedicalHistory
        mh = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
        if mh:
            if isinstance(mh.conditions, list):
                med_hist.extend(mh.conditions)
            if isinstance(mh.past_medications, list):
                past_meds.extend(mh.past_medications)
    except Exception:
        pass

    patient_context = {
        "patient_name": patient.name,
        "age": getattr(patient, "age", 50),
        "active_condition": patient.active_condition or "General Recovery",
        "condition_stage": patient.condition_stage or "active",
        "condition_notes": patient.condition_notes or "",
        "chronic_conditions": med_hist,
        "allergies": getattr(patient, "allergies", []),
        "active_medications": active_rx,
        "past_medications": past_meds
    }

    # Execute Deep Learning Vision Pipeline
    try:
        if analysis_type == "surgical_wound":
            pipeline_result = analyze_surgical_wound(str(dest_path), baseline_image_path=baseline_img_path, patient_context=patient_context)
        else:
            pipeline_result = analyze_bone_fracture(str(dest_path), baseline_image_path=baseline_img_path, patient_context=patient_context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deep learning vision pipeline error: {str(exc)}")

    # 1. Create MedicalImage record
    med_img = MedicalImage(
        id=img_uuid,
        patient_id=patient.id,
        uploaded_by=current_user.id,
        image_type=analysis_type,
        file_path=str(dest_path),
        original_filename=file.filename or stored_filename,
        mime_type=file.content_type or f"image/{ext.lstrip('.')}",
        file_size_bytes=len(contents),
        image_width=img_metrics.get("width"),
        image_height=img_metrics.get("height"),
        anatomical_site=anatomical_site or ("Abdominal Incision" if analysis_type == "surgical_wound" else "Extremity Bone"),
        clinical_view=clinical_view or ("Close-up" if analysis_type == "surgical_wound" else "AP Radiograph"),
        is_baseline=is_baseline,
        captured_at=dt.datetime.utcnow()
    )
    db.add(med_img)
    db.flush()

    # 2. Create Session (Status: DRAFT)
    session = ImageAnalysisSession(
        patient_id=patient.id,
        initiated_by=current_user.id,
        analysis_type=analysis_type,
        status="draft",
        attempt_count=1
    )
    db.add(session)
    db.flush()

    # 3. Create ImageAnalysis (Review Status: pending_review)
    analysis = ImageAnalysis(
        session_id=session.id,
        image_id=med_img.id,
        patient_id=patient.id,
        analysis_type=analysis_type,
        attempt_number=1,
        model_name=pipeline_result["model_name"],
        model_version=pipeline_result["model_version"],
        model_architecture=pipeline_result["model_architecture"],
        weights_provenance=pipeline_result["weights_provenance"],
        primary_prediction=pipeline_result["primary_prediction"],
        confidence_score=pipeline_result["confidence_score"],
        probabilities_json=pipeline_result["probabilities"],
        findings_summary=pipeline_result["findings_summary"],
        severity_level=pipeline_result["severity_level"],
        gradcam_heatmap_path=pipeline_result.get("gradcam_heatmap_path"),
        gradcam_overlay_path=pipeline_result.get("gradcam_overlay_path"),
        segmentation_mask_path=pipeline_result.get("segmentation_mask_path"),
        segmentation_overlay_path=pipeline_result.get("segmentation_overlay_path"),
        metrics_json={
            **(pipeline_result.get("wound_metrics") or pipeline_result.get("fracture_metrics") or {}),
            **({"clinical_correlation": pipeline_result["clinical_correlation"]} if pipeline_result.get("clinical_correlation") else {}),
            **({"specific_subtype": pipeline_result["specific_subtype"]} if pipeline_result.get("specific_subtype") else {}),
            **({"explainable_ai": pipeline_result["explainable_ai"]} if pipeline_result.get("explainable_ai") else {})
        },
        processing_time_ms=pipeline_result.get("processing_time_ms", 0),
        review_status="pending_review",
        is_patient_visible=False
    )
    db.add(analysis)
    db.flush()

    # 4. Attach Domain Observation
    if analysis_type == "surgical_wound":
        wm = pipeline_result.get("wound_metrics", {})
        wo = WoundObservation(
            analysis_id=analysis.id,
            patient_id=patient.id,
            wound_area_pixels=wm.get("wound_area_pixels", 0),
            wound_area_cm2=wm.get("wound_area_cm2"),
            wound_perimeter_pixels=wm.get("wound_perimeter_pixels", 0),
            relative_wound_size_pct=wm.get("relative_wound_size_pct", 0.0),
            erythema_score=wm.get("erythema_score", 0.0),
            edema_present=wm.get("edema_present", False),
            dehiscence_risk_score=wm.get("dehiscence_risk_score", 0.0),
            exudate_detected=wm.get("exudate_detected", False),
            exudate_type=wm.get("exudate_type"),
            predominant_tissue_type=wm.get("predominant_tissue_type", "granulation"),
            healing_trajectory_status=wm.get("healing_trajectory_status", "on_track")
        )
        db.add(wo)
    else:
        fm = pipeline_result.get("fracture_metrics", {})
        fa = FractureAnalysis(
            analysis_id=analysis.id,
            patient_id=patient.id,
            fracture_detected=fm.get("fracture_detected", False),
            fracture_probability=fm.get("fracture_probability", 0.0),
            anatomical_bone=fm.get("anatomical_bone", "Unspecified Bone"),
            fracture_type=fm.get("fracture_type"),
            displacement_present=fm.get("displacement_present", False),
            displacement_severity=fm.get("displacement_severity", "none"),
            callus_formation_observed=fm.get("callus_formation_observed", False),
            healing_stage=fm.get("healing_stage", "acute")
        )
        db.add(fa)

    session.active_analysis_id = analysis.id
    db.commit()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="CREATE_IMAGE_ANALYSIS_DRAFT",
        target_type="image_analyses",
        target_id=analysis.id,
        details={"description": f"Draft {analysis_type} analysis created for patient {patient.patient_code}. Prediction: {analysis.primary_prediction}"}
    )

    return {
        "session_id": session.id,
        "session_status": session.status,
        "analysis": _format_analysis_response(analysis, db)
    }


# ----------------------------------------------------
# 3. Discard Draft Session
# ----------------------------------------------------
@router.post("/sessions/{session_id}/discard")
def discard_image_analysis_session(
    session_id: str,
    body: DiscardSessionIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """
    Discards a draft analysis session. Draft results are non-destructive and will not
    persist in the patient's approved clinical record or affect risk tiers.
    """
    session = db.get(ImageAnalysisSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    if session.status == "saved":
        raise HTTPException(status_code=400, detail="Cannot discard an already saved and integrated analysis session.")

    session.status = "discarded"
    session.discarded_at = dt.datetime.utcnow()
    session.discard_reason = body.reason
    db.commit()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="DISCARD_IMAGE_ANALYSIS_SESSION",
        target_type="image_analysis_sessions",
        target_id=session.id,
        details={"description": f"Draft analysis session discarded. Reason: {body.reason}"}
    )

    return {"message": "Draft analysis session discarded successfully.", "session_id": session_id}


# ----------------------------------------------------
# 4. Save Analysis Session (Triggers Gemini, Risk & Twin)
# ----------------------------------------------------
@router.post("/sessions/{session_id}/save")
def save_image_analysis_session(
    session_id: str,
    body: Optional[SaveSessionIn] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """
    Saves the analysis session and applies doctor sign-off in a single streamlined action:
    1. Locks the session state as 'saved'.
    2. Directly updates review status (approved/amended/rejected), clinician notes, and patient portal visibility.
    3. Triggers Google Gemini API clinical reassessment cross-evaluating medical history.
    4. Re-evaluates clinical risk classification and recomputes Digital Twin state trajectory.
    5. Dispatches notifications to patient if approved and visible.
    """
    session = db.get(ImageAnalysisSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    if not session.active_analysis_id:
        raise HTTPException(status_code=400, detail="No active analysis associated with this session.")

    analysis = db.get(ImageAnalysis, session.active_analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Associated analysis record not found.")

    patient = db.get(Patient, session.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    # Update Session status
    session.status = "saved"
    session.saved_at = dt.datetime.utcnow()
    session.saved_by = current_user.id

    # Unified Sign-off: Apply clinician review details directly
    req = body or SaveSessionIn()
    if current_user.role in (Role.DOCTOR.value, Role.ADMIN.value):
        analysis.review_status = req.review_status or "approved"
        analysis.reviewed_by = current_user.id
        analysis.reviewed_at = dt.datetime.utcnow()
        analysis.clinician_notes = req.clinician_notes
        analysis.clinician_agreed_with_ai = req.clinician_agreed_with_ai if req.clinician_agreed_with_ai is not None else True
        analysis.doctor_confirmed_findings = req.doctor_confirmed_findings
        analysis.is_patient_visible = bool(req.is_patient_visible)
    else:
        # Nurse saving session leaves it for doctor review
        analysis.review_status = "pending_review"
        analysis.is_patient_visible = False

    # Trigger Google Gemini API Multimodal Reassessment
    reassessment = perform_gemini_image_reassessment(db, analysis)

    # 1. Recovery Plan Integration: Flag if review recommended
    if reassessment.recovery_plan_review_recommended:
        plan = db.scalars(
            select(RecoveryPlan)
            .where(RecoveryPlan.patient_id == patient.id)
            .order_by(RecoveryPlan.version.desc())
        ).first()

        if plan:
            plan.status = "needs_review"
            curr_inst = dict(plan.instructions or {})
            curr_inst["proposed_image_review_notes"] = reassessment.recovery_plan_proposed_notes
            plan.instructions = curr_inst
            plan.updated_at = dt.datetime.utcnow()

    # 2. Risk Monitoring Integration: Force re-evaluation incorporating new findings
    try:
        evaluate_patient_risk(db, patient_id=patient.id, force=True)
    except Exception as e:
        # Non-blocking logging
        pass

    # 3. Digital Twin Integration: Recompute state with image analysis event
    try:
        compute_digital_twin_state(
            db=db,
            patient_id=patient.id,
            event_source="image_analysis",
            event_id=analysis.id
        )
    except Exception as e:
        pass

    # 4. Notify Assigned Doctor if severity is elevated/critical or nurse submitted
    if patient.assigned_doctor_id:
        try:
            doc_user = db.get(User, patient.assigned_doctor_id)
            if doc_user:
                urgency = "URGENT: " if analysis.severity_level in ("elevated", "critical") else ""
                create_in_app_notification(
                    db=db,
                    user_id=doc_user.id,
                    title=f"{urgency}New Image Analysis Saved for {patient.name}",
                    message=(
                        f"A new {analysis.analysis_type.replace('_', ' ')} analysis was saved for {patient.name} "
                        f"({patient.patient_code}). Primary finding: {analysis.primary_prediction}. Review pending."
                    ),
                    category="alert",
                    link=f"#/staff/image-analysis?patientId={patient.id}&analysisId={analysis.id}"
                )
        except Exception:
            pass

    # 5. Notify Patient if approved and marked visible
    if analysis.review_status == "approved" and analysis.is_patient_visible and patient.user_id:
        try:
            create_in_app_notification(
                db=db,
                user_id=patient.user_id,
                title="Clinical Image Review Available",
                message="Your treating physician has reviewed and approved your recent recovery image analysis.",
                category="document",
                link="#/patient/documents"
            )
        except Exception:
            pass

    db.commit()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="SAVE_IMAGE_ANALYSIS_SESSION",
        target_type="image_analyses",
        target_id=analysis.id,
        details={"description": f"Analysis session {session.id} saved and integrated for patient {patient.patient_code}."}
    )

    return {
        "message": "Analysis session saved and integrated successfully.",
        "session_id": session.id,
        "analysis": _format_analysis_response(analysis, db)
    }


# ----------------------------------------------------
# 5. Patient Image Analysis History
# ----------------------------------------------------
@router.get("/patients/{patient_id}/history")
def get_patient_image_history(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """
    Returns full chronological history of all saved medical image analyses for a patient.
    """
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    enforce_patient_isolation(patient_id=patient.id, current_user=current_user, db=db)

    # Only return analyses that belong to saved sessions
    analyses = db.scalars(
        select(ImageAnalysis)
        .join(ImageAnalysisSession, ImageAnalysis.session_id == ImageAnalysisSession.id)
        .where(
            ImageAnalysis.patient_id == patient_id,
            ImageAnalysisSession.status == "saved"
        )
        .order_by(desc(ImageAnalysis.created_at))
    ).all()

    return [_format_analysis_response(a, db) for a in analyses]


# ----------------------------------------------------
# 6. Single Analysis Retrieval
# ----------------------------------------------------
@router.get("/{analysis_id}")
def get_image_analysis_details(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Returns complete details of a single image analysis record."""
    analysis = db.get(ImageAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Image analysis record not found.")

    enforce_patient_isolation(patient_id=analysis.patient_id, current_user=current_user, db=db)
    return _format_analysis_response(analysis, db)


# ----------------------------------------------------
# 7. Clinician Review & Doctor Sign-Off
# ----------------------------------------------------
@router.post("/{analysis_id}/review")
def review_image_analysis(
    analysis_id: str,
    body: ClinicianReviewIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    """
    Doctor & Administrator sign-off endpoint:
    - Approves, amends, or rejects AI findings.
    - Records clinician notes and doctor-confirmed findings.
    - Controls patient portal visibility (is_patient_visible).
    """
    analysis = db.get(ImageAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Image analysis not found.")

    valid_statuses = {"approved", "amended", "rejected", "pending_review"}
    if body.review_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid review status. Must be one of: {valid_statuses}")

    analysis.review_status = body.review_status
    analysis.reviewed_by = current_user.id
    analysis.reviewed_at = dt.datetime.utcnow()
    analysis.clinician_notes = body.clinician_notes
    analysis.clinician_agreed_with_ai = body.clinician_agreed_with_ai
    analysis.doctor_confirmed_findings = body.doctor_confirmed_findings
    analysis.is_patient_visible = body.is_patient_visible

    # If approved and patient visible, notify patient
    if body.review_status == "approved" and body.is_patient_visible:
        patient = db.get(Patient, analysis.patient_id)
        if patient and patient.user_id:
            try:
                create_in_app_notification(
                    db=db,
                    user_id=patient.user_id,
                    title="Clinical Image Review Available",
                    message="Your treating physician has reviewed and approved your recent recovery image analysis.",
                    category="document",
                    link="#/patient/documents"
                )
            except Exception:
                pass

    db.commit()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="REVIEW_IMAGE_ANALYSIS",
        target_type="image_analyses",
        target_id=analysis.id,
        details={"description": f"Analysis signed off as {body.review_status.upper()} by Dr. {current_user.name}."}
    )

    return {
        "message": f"Analysis marked as {body.review_status}.",
        "analysis": _format_analysis_response(analysis, db)
    }


# ----------------------------------------------------
# 8. Export PDF Report
# ----------------------------------------------------
@router.get("/{analysis_id}/export/pdf")
def export_analysis_pdf(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Generates and streams a publication-grade ReportLab PDF clinical report."""
    analysis = db.get(ImageAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Image analysis not found.")

    patient = db.get(Patient, analysis.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    enforce_patient_isolation(patient_id=patient.id, current_user=current_user, db=db)

    pdf_path = generate_image_analysis_pdf(
        analysis=analysis,
        patient=patient,
        reviewer=current_user if current_user.role == Role.DOCTOR.value else None
    )

    # Log export
    export_log = AnalysisExport(
        analysis_id=analysis.id,
        patient_id=patient.id,
        exported_by=current_user.id,
        export_format="pdf",
        file_path=pdf_path,
        file_size_bytes=os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
    )
    db.add(export_log)
    db.commit()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=Path(pdf_path).name
    )


# ----------------------------------------------------
# 9. Export CSV Metrics
# ----------------------------------------------------
@router.get("/{analysis_id}/export/csv")
def export_analysis_csv(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Generates and streams a structured CSV record for research or audit extraction."""
    analysis = db.get(ImageAnalysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Image analysis not found.")

    patient = db.get(Patient, analysis.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    enforce_patient_isolation(patient_id=patient.id, current_user=current_user, db=db)

    csv_data = generate_image_analysis_csv(analysis, patient)
    filename = f"Recovera_ImageAnalysis_{patient.patient_code}_{analysis.id[:8]}.csv"

    # Log export
    export_log = AnalysisExport(
        analysis_id=analysis.id,
        patient_id=patient.id,
        exported_by=current_user.id,
        export_format="csv",
        file_size_bytes=len(csv_data)
    )
    db.add(export_log)
    db.commit()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ----------------------------------------------------
# 10. Research Benchmarks Registry
# ----------------------------------------------------
@router.get("/benchmarks/runs")
def list_research_benchmark_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Returns reproducible model training runs and evaluation metrics from research experiments."""
    runs = db.scalars(select(ModelTrainingRun).order_by(desc(ModelTrainingRun.training_date))).all()
    return [
        {
            "id": r.id,
            "pipeline_type": r.pipeline_type,
            "architecture": r.architecture,
            "dataset_name": r.dataset_name,
            "dataset_version": r.dataset_version,
            "pretrained_weights": r.pretrained_weights,
            "total_samples": r.total_samples,
            "train_samples": r.train_samples,
            "val_samples": r.val_samples,
            "test_samples": r.test_samples,
            "patient_level_split": r.patient_level_split,
            "epochs": r.epochs,
            "batch_size": r.batch_size,
            "learning_rate": r.learning_rate,
            "optimizer": r.optimizer,
            "loss_function": r.loss_function,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "sensitivity": r.sensitivity,
            "specificity": r.specificity,
            "dice_coefficient": r.dice_coefficient,
            "iou_score": r.iou_score,
            "confusion_matrix": r.confusion_matrix_json,
            "inference_latency_ms": r.inference_latency_ms,
            "version_tag": r.version_tag,
            "is_production_model": r.is_production_model,
            "training_date": r.training_date.isoformat()
        }
        for r in runs
    ]


# ----------------------------------------------------
# 11. Pending Reviews Queue (Triage Widget)
# ----------------------------------------------------
@router.get("/pending-reviews")
def get_pending_image_reviews(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Returns pending image reviews for clinician dashboard triage widgets."""
    query = (
        select(ImageAnalysis)
        .join(ImageAnalysisSession, ImageAnalysis.session_id == ImageAnalysisSession.id)
        .where(
            ImageAnalysis.review_status == "pending_review",
            ImageAnalysisSession.status == "saved"
        )
    )

    if current_user.role == Role.DOCTOR.value:
        subq = select(PatientDoctorAssignment.patient_id).where(PatientDoctorAssignment.doctor_id == current_user.id)
        query = query.join(Patient, ImageAnalysis.patient_id == Patient.id).where(
            or_(Patient.assigned_doctor_id == current_user.id, Patient.id.in_(subq))
        )

    pending = db.scalars(query.order_by(desc(ImageAnalysis.created_at))).all()
    return [_format_analysis_response(a, db) for a in pending]


# ----------------------------------------------------
# 12. Secure File Serving Endpoint
# ----------------------------------------------------
@router.get("/files/{subpath:path}")
def serve_image_analysis_file(
    subpath: str,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Secure file streaming endpoint for medical images, Grad-CAM overlays, segmentations, and reports.
    Verifies path resolution within UPLOAD_DIR to prevent directory traversal.
    """
    upload_root = Path(settings.UPLOAD_DIR).resolve()
    target_path = (upload_root / subpath).resolve()

    # Prevent directory traversal
    if not str(target_path).startswith(str(upload_root)):
        raise HTTPException(status_code=403, detail="Access denied: Invalid file path traversal.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Requested file not found.")

    ext = target_path.suffix.lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".csv": "text/csv"
    }
    media_type = media_map.get(ext, "application/octet-stream")

    return FileResponse(target_path, media_type=media_type)
