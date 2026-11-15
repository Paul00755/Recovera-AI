# -*- coding: utf-8 -*-
"""
Recovera AI: Test Suite for Medical Image Analysis, Grad-CAM, & Deep Learning Features
Tests:
1. RBAC & Patient Access Restriction (Patients forbidden from workspace).
2. Strict Radiograph Validation (Rejecting invalid non-radiographic photos).
3. Deep Learning Vision Inference & Grad-CAM / Segmentation Generation.
4. Interactive Draft Workflow: Save vs Discard state machine.
5. Multimodal Google Gemini API Reassessment.
6. Doctor Review & Sign-Off workflow with patient visibility toggle.
7. Publication-grade PDF & CSV clinical export streaming.
8. Research benchmark model training suite verification.
"""

import io
import os
import cv2
import pytest
import numpy as np
from PIL import Image

from app.models.patient import Patient
from app.models.user import User
from app.models.image_analysis import ImageAnalysis, ImageAnalysisSession, ModelTrainingRun
from app.core.security import create_access_token
from app.core.constants import Role


def create_test_image_bytes(is_xray: bool = False, high_saturation: bool = False) -> io.BytesIO:
    """Creates a valid synthetic in-memory image buffer for testing."""
    buf = io.BytesIO()
    if is_xray:
        # Grayscale radiograph simulation
        arr = np.random.normal(40, 5, (224, 224)).clip(10, 200).astype(np.uint8)
        # Add bone diaphysis
        arr[:, 90:134] = 190
        img = Image.fromarray(arr, mode='L')
        img.save(buf, format='PNG')
    else:
        # Color skin/wound photo simulation
        if high_saturation:
            arr = np.full((224, 224, 3), (255, 60, 40), dtype=np.uint8)
        else:
            arr = np.full((224, 224, 3), (210, 180, 160), dtype=np.uint8)
            arr[80:144, 80:144] = (160, 60, 50)  # red wound bed
        img = Image.fromarray(arr, mode='RGB')
        img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


def test_patient_access_prohibited(client, db_session):
    """Verifies that patient tokens are strictly forbidden from accessing the image analysis endpoints."""
    # Find or simulate a patient user
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    if not patient or not patient.user_id:
        pytest.skip("No active patient with user account found.")

    token = create_access_token(data={"sub": patient.user_id, "role": Role.PATIENT.value})
    patient_headers = {"Authorization": f"Bearer {token}"}

    # Test patient directory
    res = client.get("/api/image-analysis/patients", headers=patient_headers)
    assert res.status_code == 403

    # Test benchmark listing
    res_bench = client.get("/api/image-analysis/benchmarks/runs", headers=patient_headers)
    assert res_bench.status_code == 403


def test_list_patients_authorized(client, doctor_headers):
    """Verifies authorized doctor can retrieve patients with active condition and baseline flags."""
    res = client.get("/api/image-analysis/patients", headers=doctor_headers)
    assert res.status_code == 200
    patients = res.json()
    assert isinstance(patients, list)
    if len(patients) > 0:
        first = patients[0]
        assert "patient_code" in first
        assert "has_baseline_wound" in first
        assert "has_baseline_xray" in first


def test_radiograph_validation_rejects_color_photo(client, doctor_headers, db_session):
    """Verifies strict medical radiograph checking rejects standard color photos when X-ray is selected."""
    patient = db_session.query(Patient).filter(Patient.patient_code == "P-103").first() or db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient available.")

    color_photo_buf = create_test_image_bytes(is_xray=False, high_saturation=True)

    files = {
        "file": ("color_photo.jpg", color_photo_buf, "image/jpeg")
    }
    data = {
        "patient_id": patient.id,
        "analysis_type": "bone_fracture_xray",  # X-ray expected, but color photo uploaded
        "anatomical_site": "Right Tibia",
        "clinical_view": "AP Radiograph",
        "is_baseline": "false"
    }

    res = client.post("/api/image-analysis/sessions", headers=doctor_headers, data=data, files=files)
    # Must reject with 400 Validation Error
    assert res.status_code == 400
    assert "Radiograph validation failed" in res.json().get("detail", "") or "Image Validation Failed" in res.json().get("detail", "")


def test_create_wound_analysis_draft_session(client, doctor_headers, db_session):
    """Tests creating a draft surgical wound analysis session with ResNet50 inference and Grad-CAM."""
    patient = db_session.query(Patient).filter(Patient.patient_code == "P-101").first() or db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient available.")

    wound_img_buf = create_test_image_bytes(is_xray=False)

    files = {
        "file": ("incision_test.jpg", wound_img_buf, "image/jpeg")
    }
    data = {
        "patient_id": patient.id,
        "analysis_type": "surgical_wound",
        "anatomical_site": "Abdominal Incision",
        "clinical_view": "Close-up",
        "is_baseline": "false"
    }

    res = client.post("/api/image-analysis/sessions", headers=doctor_headers, data=data, files=files)
    assert res.status_code == 201
    payload = res.json()

    assert "session_id" in payload
    assert payload["session_status"] == "draft"

    analysis = payload["analysis"]
    assert analysis["analysis_type"] == "surgical_wound"
    assert "ResNet50" in analysis["model_architecture"]
    assert 0.0 <= analysis["confidence_score"] <= 1.0
    assert analysis["review_status"] == "pending_review"
    assert analysis["is_patient_visible"] is False

    # Check Grad-CAM and segmentation paths
    assert analysis["artifacts"]["gradcam_overlay_url"] is not None
    assert "wound_observation" in analysis
    assert analysis["wound_observation"]["erythema_score"] >= 0.0


def test_discard_draft_session(client, doctor_headers, db_session):
    """Tests discarding a draft analysis session (non-destructive workflow)."""
    patient = db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient available.")

    wound_img_buf = create_test_image_bytes(is_xray=False)
    files = {"file": ("discard_test.jpg", wound_img_buf, "image/jpeg")}
    data = {
        "patient_id": patient.id,
        "analysis_type": "surgical_wound",
        "is_baseline": "false"
    }

    create_res = client.post("/api/image-analysis/sessions", headers=doctor_headers, data=data, files=files)
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # Discard session
    discard_res = client.post(
        f"/api/image-analysis/sessions/{session_id}/discard",
        headers=doctor_headers,
        json={"reason": "Test discard of incorrect orientation image."}
    )
    assert discard_res.status_code == 200
    assert "discarded successfully" in discard_res.json()["message"]


def test_save_session_and_gemini_reassessment(client, doctor_headers, db_session):
    """Tests saving a draft session, which triggers Gemini reassessment and system integration."""
    patient = db_session.query(Patient).filter(Patient.patient_code == "P-101").first() or db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient available.")

    wound_img_buf = create_test_image_bytes(is_xray=False)
    files = {"file": ("save_test.jpg", wound_img_buf, "image/jpeg")}
    data = {
        "patient_id": patient.id,
        "analysis_type": "surgical_wound",
        "is_baseline": "false"
    }

    create_res = client.post("/api/image-analysis/sessions", headers=doctor_headers, data=data, files=files)
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # Save session
    save_res = client.post(f"/api/image-analysis/sessions/{session_id}/save", headers=doctor_headers)
    assert save_res.status_code == 200
    save_payload = save_res.json()

    assert save_payload["analysis"]["review_status"] == "pending_review"
    assert "gemini_reassessment" in save_payload["analysis"]
    gemini = save_payload["analysis"]["gemini_reassessment"]
    assert gemini["reassessment_summary"] is not None
    assert isinstance(gemini["suggested_clinician_actions"], list)


def test_doctor_review_and_patient_visibility(client, doctor_headers, db_session):
    """Tests doctor reviewing, signing off, and toggling patient visibility."""
    analysis = db_session.query(ImageAnalysis).first()
    if not analysis:
        pytest.skip("No saved image analysis available.")

    review_data = {
        "review_status": "approved",
        "clinician_notes": "Post-operative incision healing favorably. Clean dressing applied.",
        "clinician_agreed_with_ai": True,
        "doctor_confirmed_findings": "Normal healing with active granulation tissue.",
        "is_patient_visible": True
    }

    res = client.post(f"/api/image-analysis/{analysis.id}/review", headers=doctor_headers, json=review_data)
    assert res.status_code == 200
    updated = res.json()["analysis"]
    assert updated["review_status"] == "approved"
    assert updated["is_patient_visible"] is True
    assert updated["clinician_notes"] == review_data["clinician_notes"]


def test_export_pdf_and_csv(client, doctor_headers, db_session):
    """Tests ReportLab PDF clinical report generation and structured CSV export."""
    analysis = db_session.query(ImageAnalysis).first()
    if not analysis:
        pytest.skip("No saved image analysis available.")

    # 1. PDF Export
    pdf_res = client.get(f"/api/image-analysis/{analysis.id}/export/pdf", headers=doctor_headers)
    assert pdf_res.status_code == 200
    assert pdf_res.headers.get("content-type") == "application/pdf"
    assert len(pdf_res.content) > 1000  # valid PDF bytes

    # 2. CSV Export
    csv_res = client.get(f"/api/image-analysis/{analysis.id}/export/csv", headers=doctor_headers)
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type")
    assert "Analysis_ID" in csv_res.text


def test_research_benchmarks_runs(client, doctor_headers):
    """Tests retrieving reproducible deep learning model training benchmark records."""
    res = client.get("/api/image-analysis/benchmarks/runs", headers=doctor_headers)
    assert res.status_code == 200
    runs = res.json()
    assert isinstance(runs, list)
    assert len(runs) >= 4  # ResNet50, DenseNet121, U-Net++, etc.
    first_run = runs[0]
    assert "architecture" in first_run
    assert "accuracy" in first_run
    assert "sensitivity" in first_run
    assert "specificity" in first_run
    assert first_run["patient_level_split"] is True
