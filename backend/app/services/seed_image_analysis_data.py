"""
Recovera AI: 5-Patient Experimental Cohort Seeder for Image Analysis
Seeds:
1. Patient P-101 (Rajesh Kumar): Post-op Laparoscopic Incision (Baseline Day 1 & Follow-up Day 7).
2. Patient P-102 (Priya Sharma): Diabetic Surgical Wound with Periwound Erythema Alert.
3. Patient P-103 (Madhavan S): Right Tibial Shaft Fracture (Acute vs Day 14 Callus Formation).
4. Patient P-104 (Ananya Patel): Left Distal Radius Fracture (Acute Radiograph with Cortical Disruption).
5. Patient P-105 (Suresh Menon): Post-Knee Arthroplasty Midline Surgical Incision (Longitudinal).

Ensures realistic images, Grad-CAM heatmaps, segmentations, and Gemini clinical reassessments exist in the database.
"""

import os
import cv2
import uuid
import datetime as dt
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.config import settings
from app.models.patient import Patient
from app.models.user import User
from app.models.image_analysis import (
    MedicalImage,
    ImageAnalysisSession,
    ImageAnalysis,
    WoundObservation,
    FractureAnalysis,
    GeminiReassessment,
    ModelTrainingRun
)
from app.services.medical_vision_service import analyze_surgical_wound, analyze_bone_fracture
from app.services.image_gemini_reassessment_service import perform_gemini_image_reassessment
from app.ml.image_training_pipeline import seed_research_benchmark_suite


def _generate_synthetic_wound_image(output_path: str, size=(480, 480), erythema: bool = False, healed: bool = False):
    """Generates a realistic clinical wound test image with skin gradients, incision margins, and optional erythema."""
    h, w = size
    # Skin tone background with natural variation
    skin_bgr = np.full((h, w, 3), (170, 190, 225), dtype=np.uint8)
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    skin_bgr = np.clip(skin_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    center_x, center_y = w // 2, h // 2

    # Erythema halo if flagged
    if erythema:
        cv2.circle(skin_bgr, (center_x, center_y), 110, (110, 130, 235), -1)
        skin_bgr = cv2.GaussianBlur(skin_bgr, (51, 51), 0)

    # Wound bed
    wound_color = (60, 70, 160) if not healed else (120, 140, 200)
    pts = np.array([
        [center_x - 30, center_y - 90],
        [center_x + 10, center_y - 40],
        [center_x + 25, center_y + 30],
        [center_x - 5, center_y + 90],
        [center_x - 35, center_y + 40],
        [center_x - 20, center_y - 30]
    ], np.int32)
    cv2.fillPoly(skin_bgr, [pts], wound_color)

    # Incision line and suture tick marks
    cv2.polylines(skin_bgr, [pts], isClosed=True, color=(30, 40, 110), thickness=2)
    for y_offset in range(-70, 80, 25):
        suture_y = center_y + y_offset
        cv2.line(skin_bgr, (center_x - 45, suture_y), (center_x + 35, suture_y), (40, 40, 60), 2)
        cv2.circle(skin_bgr, (center_x - 45, suture_y), 3, (20, 20, 40), -1)
        cv2.circle(skin_bgr, (center_x + 35, suture_y), 3, (20, 20, 40), -1)

    skin_bgr = cv2.GaussianBlur(skin_bgr, (3, 3), 0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, skin_bgr)
    return output_path


def _generate_synthetic_xray_image(output_path: str, size=(480, 480), fracture: bool = True, callus: bool = False):
    """Generates a realistic radiographic test image with bone diaphysis, cortex, and optional fracture gap."""
    h, w = size
    # Dark radiograph background (soft tissue / air)
    xray = np.random.normal(30, 6, (h, w)).clip(10, 60).astype(np.uint8)

    # Soft tissue envelope
    cv2.rectangle(xray, (w // 4 - 30, 0), (3 * w // 4 + 30, h), 75, -1)

    # Cortical bone diaphysis (radiopaque white/light gray)
    bone_left = w // 2 - 45
    bone_right = w // 2 + 45
    cv2.rectangle(xray, (bone_left, 0), (bone_right, h), 200, -1)

    # Medullary cavity (slightly darker)
    cv2.rectangle(xray, (bone_left + 14, 0), (bone_right - 14, h), 140, -1)

    # Fracture radiolucency / gap if flagged
    mid_y = h // 2
    if fracture:
        if callus:
            # Callus bridging (fluffy periosteal new bone)
            cv2.ellipse(xray, (w // 2, mid_y), (58, 28), 0, 0, 360, 175, -1)
            # Faint remnant fracture line
            cv2.line(xray, (bone_left - 5, mid_y), (bone_right + 5, mid_y), 110, 2)
        else:
            # Acute sharp fracture gap with step-off
            pts = np.array([
                [bone_left - 10, mid_y - 6],
                [w // 2 - 5, mid_y + 4],
                [bone_right + 10, mid_y - 2]
            ], np.int32)
            cv2.polylines(xray, [pts], isClosed=False, color=20, thickness=5)
            # Slight cortical offset
            xray[mid_y:, bone_left:bone_right] = np.roll(xray[mid_y:, bone_left:bone_right], shift=4, axis=1)

    xray = cv2.GaussianBlur(xray, (3, 3), 0)
    # Convert to 3-channel grayscale BGR
    xray_bgr = cv2.cvtColor(xray, cv2.COLOR_GRAY2BGR)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, xray_bgr)
    return output_path


def seed_image_analysis_cohort(db: Session):
    """
    Seeds baseline and follow-up clinical image analysis records for the 5 research patients.
    Also ensures the benchmark suite runs are seeded.
    """
    # 1. Ensure benchmark runs exist
    seed_research_benchmark_suite()

    # 2. Find treating doctors and nurses
    doctor = db.scalars(select(User).where(User.role == "doctor", User.active == True)).first()
    nurse = db.scalars(select(User).where(User.role == "nurse", User.active == True)).first()
    admin = db.scalars(select(User).where(User.role == "admin", User.active == True)).first()

    creator_id = doctor.id if doctor else (admin.id if admin else str(uuid.uuid4()))
    nurse_id = nurse.id if nurse else creator_id

    # 3. Patient definitions & clinical image scenarios
    cohort_specs = [
        {
            "code": "P-101",
            "name": "Rajesh Kumar",
            "condition": "Post-op Laparoscopic Appendectomy",
            "type": "surgical_wound",
            "site": "Lower Right Abdominal Incision",
            "view": "Close-up Tangential",
            "baseline_date": dt.datetime.utcnow() - dt.timedelta(days=7),
            "followup_date": dt.datetime.utcnow() - dt.timedelta(days=1),
            "has_followup": True
        },
        {
            "code": "P-102",
            "name": "Priya Sharma",
            "condition": "Type 2 Diabetes Mellitus with Post-Viral Asthenia",
            "type": "surgical_wound",
            "site": "Left Foot Surgical Debridement Site",
            "view": "Dorsal Close-up",
            "baseline_date": dt.datetime.utcnow() - dt.timedelta(days=4),
            "followup_date": None,
            "has_followup": False,
            "erythema": True
        },
        {
            "code": "P-103",
            "name": "Madhavan S",
            "condition": "Right Tibial Shaft Fracture",
            "type": "bone_fracture_xray",
            "site": "Right Tibia / Fibula",
            "view": "AP & Lateral Radiograph",
            "baseline_date": dt.datetime.utcnow() - dt.timedelta(days=14),
            "followup_date": dt.datetime.utcnow() - dt.timedelta(days=1),
            "has_followup": True,
            "callus_followup": True
        },
        {
            "code": "P-104",
            "name": "Ananya Patel",
            "condition": "Left Distal Radius (Colles) Fracture",
            "type": "bone_fracture_xray",
            "site": "Left Distal Radius",
            "view": "PA Wrist Radiograph",
            "baseline_date": dt.datetime.utcnow() - dt.timedelta(days=3),
            "followup_date": None,
            "has_followup": False
        },
        {
            "code": "P-105",
            "name": "Suresh Menon",
            "condition": "Post-Knee Arthroplasty Recovery",
            "type": "surgical_wound",
            "site": "Anterior Midline Right Knee",
            "view": "Anterior Longitudinal",
            "baseline_date": dt.datetime.utcnow() - dt.timedelta(days=10),
            "followup_date": dt.datetime.utcnow() - dt.timedelta(days=2),
            "has_followup": True
        }
    ]

    images_base_dir = Path(settings.UPLOAD_DIR) / "medical_images"
    images_base_dir.mkdir(parents=True, exist_ok=True)

    for spec in cohort_specs:
        patient = db.scalars(select(Patient).where(Patient.patient_code == spec["code"])).first()
        if not patient:
            continue

        # Check if already seeded
        existing_imgs = db.scalars(select(MedicalImage).where(MedicalImage.patient_id == patient.id)).all()
        if len(existing_imgs) >= 1:
            continue

        # ----------------------------------------------------
        # Baseline Image & Analysis
        # ----------------------------------------------------
        b_img_id = str(uuid.uuid4())
        if spec["type"] == "surgical_wound":
            b_path = str(images_base_dir / "wounds" / f"{spec['code']}_baseline_{b_img_id[:8]}.jpg")
            _generate_synthetic_wound_image(b_path, erythema=spec.get("erythema", False), healed=False)
            pipeline_result = analyze_surgical_wound(b_path)
        else:
            b_path = str(images_base_dir / "xrays" / f"{spec['code']}_baseline_{b_img_id[:8]}.png")
            _generate_synthetic_xray_image(b_path, fracture=True, callus=False)
            pipeline_result = analyze_bone_fracture(b_path)

        b_med_img = MedicalImage(
            id=b_img_id,
            patient_id=patient.id,
            uploaded_by=nurse_id,
            image_type=spec["type"],
            file_path=b_path,
            original_filename=f"{spec['code']}_baseline.jpg" if spec["type"] == "surgical_wound" else f"{spec['code']}_baseline.png",
            mime_type="image/jpeg" if spec["type"] == "surgical_wound" else "image/png",
            file_size_bytes=os.path.getsize(b_path),
            image_width=480,
            image_height=480,
            anatomical_site=spec["site"],
            clinical_view=spec["view"],
            is_baseline=True,
            captured_at=spec["baseline_date"],
            created_at=spec["baseline_date"]
        )
        db.add(b_med_img)
        db.flush()

        # Create Session (Saved)
        b_session = ImageAnalysisSession(
            patient_id=patient.id,
            initiated_by=nurse_id,
            analysis_type=spec["type"],
            status="saved",
            attempt_count=1,
            saved_at=spec["baseline_date"],
            saved_by=doctor.id if doctor else creator_id,
            created_at=spec["baseline_date"]
        )
        db.add(b_session)
        db.flush()

        # Create Analysis Record
        b_analysis = ImageAnalysis(
            session_id=b_session.id,
            image_id=b_med_img.id,
            patient_id=patient.id,
            analysis_type=spec["type"],
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
            metrics_json=pipeline_result.get("wound_metrics") or pipeline_result.get("fracture_metrics") or {},
            processing_time_ms=pipeline_result.get("processing_time_ms", 120),
            review_status="approved",
            reviewed_by=doctor.id if doctor else creator_id,
            reviewed_at=spec["baseline_date"] + dt.timedelta(hours=2),
            clinician_notes="Baseline clinical submission verified. Anatomical alignment and baseline characteristics confirmed.",
            clinician_agreed_with_ai=True,
            doctor_confirmed_findings=pipeline_result["findings_summary"],
            is_patient_visible=True,
            created_at=spec["baseline_date"]
        )
        db.add(b_analysis)
        db.flush()
        b_session.active_analysis_id = b_analysis.id

        # Add Observation
        if spec["type"] == "surgical_wound":
            wm = pipeline_result.get("wound_metrics", {})
            b_wo = WoundObservation(
                analysis_id=b_analysis.id,
                patient_id=patient.id,
                wound_area_pixels=wm.get("wound_area_pixels", 2400),
                wound_area_cm2=wm.get("wound_area_cm2", 4.2),
                wound_perimeter_pixels=wm.get("wound_perimeter_pixels", 260),
                relative_wound_size_pct=wm.get("relative_wound_size_pct", 1.8),
                erythema_score=wm.get("erythema_score", 45.0 if spec.get("erythema") else 18.0),
                edema_present=wm.get("edema_present", False),
                dehiscence_risk_score=wm.get("dehiscence_risk_score", 12.0),
                exudate_detected=wm.get("exudate_detected", False),
                predominant_tissue_type=wm.get("predominant_tissue_type", "granulation"),
                healing_trajectory_status="on_track",
                created_at=spec["baseline_date"]
            )
            db.add(b_wo)
        else:
            fm = pipeline_result.get("fracture_metrics", {})
            b_fa = FractureAnalysis(
                analysis_id=b_analysis.id,
                patient_id=patient.id,
                fracture_detected=fm.get("fracture_detected", True),
                fracture_probability=fm.get("fracture_probability", 0.92),
                anatomical_bone=fm.get("anatomical_bone", spec["site"]),
                fracture_type="transverse",
                displacement_present=fm.get("displacement_present", False),
                displacement_severity=fm.get("displacement_severity", "minimal"),
                callus_formation_observed=False,
                healing_stage="acute",
                created_at=spec["baseline_date"]
            )
            db.add(b_fa)

        # Run Gemini Reassessment
        perform_gemini_image_reassessment(db, b_analysis)

        # ----------------------------------------------------
        # Follow-up Image & Longitudinal Comparison
        # ----------------------------------------------------
        if spec.get("has_followup") and spec.get("followup_date"):
            f_img_id = str(uuid.uuid4())
            if spec["type"] == "surgical_wound":
                f_path = str(images_base_dir / "wounds" / f"{spec['code']}_followup_{f_img_id[:8]}.jpg")
                _generate_synthetic_wound_image(f_path, erythema=False, healed=True)
                f_pipeline_result = analyze_surgical_wound(f_path, baseline_image_path=b_path)
            else:
                f_path = str(images_base_dir / "xrays" / f"{spec['code']}_followup_{f_img_id[:8]}.png")
                _generate_synthetic_xray_image(f_path, fracture=True, callus=spec.get("callus_followup", False))
                f_pipeline_result = analyze_bone_fracture(f_path, baseline_image_path=b_path)

            f_med_img = MedicalImage(
                id=f_img_id,
                patient_id=patient.id,
                uploaded_by=nurse_id,
                image_type=spec["type"],
                file_path=f_path,
                original_filename=f"{spec['code']}_followup.jpg" if spec["type"] == "surgical_wound" else f"{spec['code']}_followup.png",
                mime_type="image/jpeg" if spec["type"] == "surgical_wound" else "image/png",
                file_size_bytes=os.path.getsize(f_path),
                image_width=480,
                image_height=480,
                anatomical_site=spec["site"],
                clinical_view=spec["view"],
                is_baseline=False,
                captured_at=spec["followup_date"],
                created_at=spec["followup_date"]
            )
            db.add(f_med_img)
            db.flush()

            f_session = ImageAnalysisSession(
                patient_id=patient.id,
                initiated_by=nurse_id,
                analysis_type=spec["type"],
                status="saved",
                attempt_count=1,
                saved_at=spec["followup_date"],
                saved_by=doctor.id if doctor else creator_id,
                created_at=spec["followup_date"]
            )
            db.add(f_session)
            db.flush()

            f_analysis = ImageAnalysis(
                session_id=f_session.id,
                image_id=f_med_img.id,
                patient_id=patient.id,
                analysis_type=spec["type"],
                attempt_number=1,
                model_name=f_pipeline_result["model_name"],
                model_version=f_pipeline_result["model_version"],
                model_architecture=f_pipeline_result["model_architecture"],
                weights_provenance=f_pipeline_result["weights_provenance"],
                primary_prediction=f_pipeline_result["primary_prediction"],
                confidence_score=f_pipeline_result["confidence_score"],
                probabilities_json=f_pipeline_result["probabilities"],
                findings_summary=f_pipeline_result["findings_summary"],
                severity_level=f_pipeline_result["severity_level"],
                gradcam_heatmap_path=f_pipeline_result.get("gradcam_heatmap_path"),
                gradcam_overlay_path=f_pipeline_result.get("gradcam_overlay_path"),
                segmentation_mask_path=f_pipeline_result.get("segmentation_mask_path"),
                segmentation_overlay_path=f_pipeline_result.get("segmentation_overlay_path"),
                metrics_json=f_pipeline_result.get("wound_metrics") or f_pipeline_result.get("fracture_metrics") or {},
                processing_time_ms=f_pipeline_result.get("processing_time_ms", 115),
                review_status="approved",
                reviewed_by=doctor.id if doctor else creator_id,
                reviewed_at=spec["followup_date"] + dt.timedelta(hours=1),
                clinician_notes="Sequential follow-up review: Significant positive recovery progression noted against baseline.",
                clinician_agreed_with_ai=True,
                doctor_confirmed_findings=f_pipeline_result["findings_summary"],
                is_patient_visible=True,
                created_at=spec["followup_date"]
            )
            db.add(f_analysis)
            db.flush()
            f_session.active_analysis_id = f_analysis.id

            if spec["type"] == "surgical_wound":
                f_wm = f_pipeline_result.get("wound_metrics", {})
                f_wo = WoundObservation(
                    analysis_id=f_analysis.id,
                    patient_id=patient.id,
                    wound_area_pixels=f_wm.get("wound_area_pixels", 1100),
                    wound_area_cm2=f_wm.get("wound_area_cm2", 2.1),
                    wound_perimeter_pixels=f_wm.get("wound_perimeter_pixels", 180),
                    relative_wound_size_pct=f_wm.get("relative_wound_size_pct", 0.9),
                    erythema_score=f_wm.get("erythema_score", 12.0),
                    edema_present=False,
                    dehiscence_risk_score=5.0,
                    exudate_detected=False,
                    predominant_tissue_type="epithelializing",
                    healing_trajectory_status="on_track",
                    created_at=spec["followup_date"]
                )
                db.add(f_wo)
            else:
                f_fm = f_pipeline_result.get("fracture_metrics", {})
                f_fa = FractureAnalysis(
                    analysis_id=f_analysis.id,
                    patient_id=patient.id,
                    fracture_detected=f_fm.get("fracture_detected", True),
                    fracture_probability=f_fm.get("fracture_probability", 0.78),
                    anatomical_bone=f_fm.get("anatomical_bone", spec["site"]),
                    fracture_type="consolidating",
                    displacement_present=False,
                    displacement_severity="none",
                    callus_formation_observed=True,
                    healing_stage="early_callus",
                    created_at=spec["followup_date"]
                )
                db.add(f_fa)

            perform_gemini_image_reassessment(db, f_analysis)

    db.commit()
    return True
