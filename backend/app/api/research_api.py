"""
Recovera AI – Research, Deep Learning & Medical Document Benchmark API
Exposes endpoints for:
- Model registry, training configurations, and comprehensive evaluation metrics
- Retraining pipeline execution with live validation
- 3-Way OCR + NLP comparative benchmark (PaddleOCR vs Deep Vision vs Hybrid OCR+NLP)
- 5-Patient controlled experimental cohort tracking across 10 clinical dimensions
- CSV/Report export for academic research and publication
"""

import io
import csv
import logging
import datetime as dt
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_db
from app.core.constants import Role
from app.models.user import User
from app.models.patient import Patient
from app.models.clinical_document import ClinicalDocument, OCRExtraction
from app.models.medication import Medication, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.models.digital_twin import DigitalTwinState, RecoveryTrajectoryMilestone
from app.models.risk_history import RiskClassificationHistory
from app.api.deps import get_current_user, require_role
from app.ml.training_pipeline import ModelTrainingPipeline, FEATURE_NAMES
from app.ml.ocr_evaluation_service import MedicalOCREvaluationService
from app.ml.prediction_engine import (
    predict_patient_multi_model,
    reload_model_cache,
    get_cached_models
)
from app.services.seed_experimental_data import seed_five_patient_experimental_cohort

logger = logging.getLogger("recovera.research_api")

router = APIRouter(prefix="/api/research", tags=["Academic Research & Model Evaluation"])


class RetrainRequest(BaseModel):
    version_tag: Optional[str] = Field("v1.1", description="New model version identifier")
    n_patients: Optional[int] = Field(500, ge=100, le=2000, description="Synthetic + grounded patient cohort size")


@router.get("/models/overview")
def get_models_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns full research evaluation metrics for all trained models:
    Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC,
    Confusion Matrix, Latency, and Feature Importance Rankings.
    """
    registry = ModelTrainingPipeline.load_active_registry()
    if not registry:
        registry = ModelTrainingPipeline.train_and_evaluate_all_models(version_tag="v1.0")

    return registry


@router.post("/models/retrain")
def retrain_models(
    req: RetrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    """
    Triggers automated training pipeline across all candidate models,
    evaluates on a held-out patient-isolated test set, versions artifacts, and reloads inference cache.
    """
    tag = req.version_tag or f"v{int(time.time())}"
    updated_registry = ModelTrainingPipeline.train_and_evaluate_all_models(
        version_tag=tag,
        n_patients=req.n_patients or 500
    )
    reload_model_cache()
    return {
        "status": "success",
        "message": f"Successfully retrained and published models under version '{tag}'",
        "registry": updated_registry
    }


@router.get("/ocr-nlp/benchmark")
def get_ocr_nlp_benchmark(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes and returns the 3-Way Comparative Medical Document Extraction Benchmark:
    Approach 1: PaddleOCR Baseline (Regex Parser)
    Approach 2: Deep Vision OCR Pipeline (CLAHE + Morphological Binarization)
    Approach 3: Hybrid OCR + NLP Pipeline (Top Performer: Canonical Entity Normalizer)

    Outputs CER, WER, Precision, Recall, F1, Exact Match, and Latency.
    """
    results = MedicalOCREvaluationService.evaluate_benchmark_suite()
    return results


@router.get("/experimental/cohort")
def get_five_patient_experimental_cohort(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the comprehensive 10-dimension experimental comparison matrix
    for the five controlled research patient records (P-101 through P-105).
    """
    # Ensure cohort exists
    existing = db.scalars(select(Patient).where(Patient.patient_code.in_(["P-101", "P-102", "P-103", "P-104", "P-105"]))).all()
    if len(existing) < 5:
        seed_five_patient_experimental_cohort(db, force_reset=False)
        existing = db.scalars(select(Patient).where(Patient.patient_code.in_(["P-101", "P-102", "P-103", "P-104", "P-105"]))).all()

    sorted_patients = sorted(existing, key=lambda p: p.patient_code)
    cohort_matrix = []

    for patient in sorted_patients:
        pid = str(patient.id)

        # 1. Document Extraction Status & Results
        doc = db.scalar(
            select(ClinicalDocument)
            .where(ClinicalDocument.patient_id == pid)
            .order_by(ClinicalDocument.created_at.desc())
            .limit(1)
        )
        ocr = db.scalar(
            select(OCRExtraction)
            .where(OCRExtraction.document_id == doc.id)
            .order_by(OCRExtraction.created_at.desc())
            .limit(1)
        ) if doc else None

        doc_status = doc.status if doc else "No Document"
        ocr_text_preview = (ocr.cleaned_text[:140] + "...") if (ocr and ocr.cleaned_text) else "N/A"
        nlp_meds = doc.structured_data.get("medications", []) if (doc and doc.structured_data) else []

        # 2. Medication Adherence
        occs = db.scalars(select(MedicationOccurrence).where(MedicationOccurrence.patient_id == pid)).all()
        tot_occ = len(occs)
        taken_occ = sum(1 for o in occs if o.state == "taken")
        adh_pct = round((taken_occ / max(1, tot_occ)) * 100.0, 1) if tot_occ > 0 else None

        # 3. Symptom Progression & Pain Trajectory
        checkins = db.scalars(
            select(DailyCheckIn)
            .where(DailyCheckIn.patient_id == pid)
            .order_by(DailyCheckIn.checkin_date.asc())
        ).all()
        pain_trajectory = [{"date": c.checkin_date.isoformat(), "pain": c.pain_level, "fever": c.fever} for c in checkins]
        latest_pain = checkins[-1].pain_level if checkins else None
        has_fever = any(c.fever for c in checkins)

        # 4. Digital Twin State
        dt_state = db.scalar(select(DigitalTwinState).where(DigitalTwinState.patient_id == pid))
        recovery_score = dt_state.current_recovery_score if dt_state else 100.0
        deviation_tier = dt_state.deviation_tier if dt_state else "normal"
        comp_deviation = dt_state.composite_deviation if dt_state else 0.0

        # 5. Risk Assessment (Explainable Green/Yellow/Red)
        risk_hist = db.scalar(
            select(RiskClassificationHistory)
            .where(RiskClassificationHistory.patient_id == pid)
            .order_by(RiskClassificationHistory.evaluated_at.desc())
            .limit(1)
        )
        risk_level = risk_hist.level if risk_hist else "insufficient_data"
        risk_reasons = risk_hist.explainability_reasons if risk_hist else ["No assessment available"]

        # 6. ML Model Multi-Predictions (LogReg, Random Forest, XGBoost, LSTM)
        ml_preds = predict_patient_multi_model(db, pid)

        cohort_matrix.append({
            "patient_id": pid,
            "patient_code": patient.patient_code,
            "name": patient.user.name if patient.user else patient.patient_code,
            "gender": patient.gender,
            "condition": patient.active_condition,
            "condition_stage": patient.condition_stage,
            "document_extraction_status": doc_status,
            "document_filename": doc.filename if doc else None,
            "ocr_text_preview": ocr_text_preview,
            "nlp_extracted_medications_count": len(nlp_meds),
            "nlp_medications": nlp_meds,
            "medication_adherence_pct": adh_pct,
            "adherence_pct": adh_pct,
            "total_occurrences": tot_occ,
            "symptom_progression": pain_trajectory,
            "latest_pain_level": latest_pain,
            "has_active_fever": has_fever,
            "digital_twin_recovery_score": recovery_score,
            "digital_twin_deviation_tier": deviation_tier,
            "composite_deviation": comp_deviation,
            "risk_classification": risk_level.upper(),
            "risk_reasons": risk_reasons,
            "ml_model_predictions": ml_preds.get("predictions", []),
            "key_drivers": ml_preds.get("key_drivers", [])
        })

    return {
        "cohort_size": len(cohort_matrix),
        "evaluated_at": dt.datetime.utcnow().isoformat(),
        "patients": cohort_matrix
    }


@router.post("/experimental/reset")
def reset_five_patient_experimental_cohort(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    """Resets the five experimental patient records to original ground-truth states."""
    res = seed_five_patient_experimental_cohort(db, force_reset=True)
    return res


@router.get("/export/csv")
def export_research_data_csv(
    data_type: str = Query("experimental_cohort", pattern="^(experimental_cohort|ocr_benchmark|model_metrics)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Exports research tables as standardized CSV files for academic publication."""
    output = io.StringIO()
    writer = csv.writer(output)

    if data_type == "experimental_cohort":
        data = get_five_patient_experimental_cohort(db, current_user)
        patients = data.get("patients", [])
        writer.writerow([
            "Patient ID", "Code", "Name", "Condition", "Stage", "Adherence %",
            "Latest Pain", "Fever", "Digital Twin Score", "Deviation Tier",
            "Risk Classification", "XGBoost Prob %", "Random Forest Prob %", "LSTM Projected Score"
        ])
        for p in patients:
            preds = {m["model_id"]: m for m in p.get("ml_model_predictions", [])}
            xgb_p = preds.get("xgboost", {}).get("probability", "N/A")
            rf_p = preds.get("random_forest", {}).get("probability", "N/A")
            lstm_p = preds.get("lstm_trajectory", {}).get("predicted_value", "N/A")

            writer.writerow([
                p["patient_id"],
                p["patient_code"],
                p["name"],
                p["condition"],
                p["condition_stage"],
                p["medication_adherence_pct"] or "N/A",
                p["latest_pain_level"] if p["latest_pain_level"] is not None else "N/A",
                "Yes" if p["has_active_fever"] else "No",
                p["digital_twin_recovery_score"],
                p["digital_twin_deviation_tier"],
                p["risk_classification"],
                xgb_p,
                rf_p,
                lstm_p
            ])

    elif data_type == "ocr_benchmark":
        bench = MedicalOCREvaluationService.evaluate_benchmark_suite()
        summary = bench.get("comparison_summary", [])
        writer.writerow(["Approach ID", "Name", "CER (%)", "WER (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Exact Match (%)", "Mean Latency (ms)", "Strengths"])
        for s in summary:
            writer.writerow([
                s["approach_id"],
                s["name"],
                s["cer"],
                s["wer"],
                s["precision"],
                s["recall"],
                s["f1_score"],
                s["exact_match"],
                s["mean_latency_ms"],
                s["strengths"]
            ])

    elif data_type == "model_metrics":
        reg = ModelTrainingPipeline.load_active_registry()
        models = reg.get("models", [])
        writer.writerow(["Model ID", "Model Name", "Architecture", "Accuracy (%)", "Precision (%)", "Recall/Sensitivity (%)", "Specificity (%)", "F1 Score (%)", "ROC-AUC", "PR-AUC", "Latency (s)"])
        for m in models:
            met = m.get("metrics", {})
            writer.writerow([
                m["model_id"],
                m["model_name"],
                m["architecture_type"],
                met.get("accuracy", "N/A"),
                met.get("precision", "N/A"),
                met.get("recall_sensitivity", "N/A"),
                met.get("specificity", "N/A"),
                met.get("f1_score", "N/A"),
                met.get("roc_auc", "N/A"),
                met.get("pr_auc", "N/A"),
                m.get("training_time_sec", "N/A")
            ])

    content = output.getvalue()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=recovera_{data_type}.csv"}
    )
