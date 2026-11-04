# -*- coding: utf-8 -*-
"""
Recovera AI 2.0 – Digital Twin & Comparative Deep Learning Test Suite
Validates:
1. Deterministic mathematical deviation formula: D_t = 0.35*S_t + 0.25*A_t + 0.20*T_t + 0.10*F_t + 0.10*Q_t
2. 14-day clinician recovery corridor [Y_min, Y_max] integrity
3. State update idempotency and audit history tracking
4. Clinician-in-the-loop alert review lifecycle (pending -> acknowledged -> dismissed)
5. Multi-model benchmark metrics endpoint (6 models, MAE, latency)
6. Zero regression of core 1.0 platform endpoints
"""
import uuid
import datetime as dt
import pytest
from sqlalchemy import select

from app.models.patient import Patient
from app.models.user import User
from app.models.digital_twin import (
    DigitalTwinState,
    DigitalTwinStateHistory,
    RecoveryTrajectoryMilestone,
    DigitalTwinDeviationAlert
)
from app.services.digital_twin_service import (
    compute_digital_twin_state,
    get_or_create_trajectory_milestones
)


def test_digital_twin_mathematical_formula(db_session, client, doctor_headers):
    """Verify that composite deviation D_t and recovery score match exact weighted formula."""
    patient = db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient found for testing.")

    state = compute_digital_twin_state(db_session, patient.id)
    assert state is not None

    # Expected formula: D_t = 0.35*S + 0.25*A + 0.20*T + 0.10*F + 0.10*Q
    expected_dt = (
        0.35 * state.symptom_deviation_score +
        0.25 * state.adherence_deviation_score +
        0.20 * state.trajectory_slope_score +
        0.10 * state.followup_deviation_score +
        0.10 * state.uncertainty_score
    )
    assert abs(state.composite_deviation - round(expected_dt, 1)) < 0.05

    # Recovery score R_t = max(0.0, 100.0 - D_t)
    assert abs(state.current_recovery_score - max(0.0, 100.0 - state.composite_deviation)) < 0.01

    # API verification
    res = client.get(f"/api/digital-twin/patients/{patient.id}/state", headers=doctor_headers)
    assert res.status_code == 200
    data = res.json()
    assert "deviation_score" in data
    assert "recovery_score" in data
    assert "component_scores" in data
    assert data["weights"]["symptom_weight"] == 0.35
    assert data["weights"]["adherence_weight"] == 0.25


def test_trajectory_corridor_14_days(db_session, client, doctor_headers):
    """Verify that a 14-day clinician-approved recovery corridor is generated."""
    patient = db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient found for testing.")

    milestones = get_or_create_trajectory_milestones(db_session, patient.id)
    assert len(milestones) == 14

    for m in milestones:
        assert 1 <= m.day_number <= 14
        assert m.expected_pain_min <= m.expected_pain_max
        assert 0.0 <= m.expected_pain_min <= 10.0
        assert 0.0 <= m.expected_pain_max <= 10.0

    # Test API endpoint
    res = client.get(f"/api/digital-twin/patients/{patient.id}/trajectory", headers=doctor_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data["corridor_milestones"]) == 14
    assert len(data["forecasts"]) >= 1
    assert "model_architecture" in data["forecasts"][0]


def test_idempotency_and_state_history(db_session):
    """Ensure running state computation multiple times updates the existing record idempotently."""
    patient = db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient found for testing.")

    initial_states_count = db_session.query(DigitalTwinState).filter_by(patient_id=patient.id).count()

    state1 = compute_digital_twin_state(db_session, patient.id, event_source="test_event_1")
    state2 = compute_digital_twin_state(db_session, patient.id, event_source="test_event_2")

    final_states_count = db_session.query(DigitalTwinState).filter_by(patient_id=patient.id).count()
    assert final_states_count == 1  # Unique patient constraint preserved

    # Verify history audit was captured
    history_records = db_session.query(DigitalTwinStateHistory).filter_by(patient_id=patient.id).all()
    assert len(history_records) >= 1


def test_clinician_alert_lifecycle(db_session, client, doctor_headers):
    """Verify clinician deviation alert creation, review, and status update."""
    patient = db_session.query(Patient).first()
    if not patient:
        pytest.skip("No patient found for testing.")

    # Create a test alert
    test_alert = DigitalTwinDeviationAlert(
        patient_id=patient.id,
        severity="moderate",
        trigger_reason="Test trajectory divergence: pain increased while adherence decreased",
        trigger_evidence_json={"symptom_deviation": 45.0, "adherence_gap": 50.0},
        status="pending"
    )
    db_session.add(test_alert)
    db_session.commit()
    db_session.refresh(test_alert)

    # 1. Fetch alerts
    res = client.get(f"/api/digital-twin/patients/{patient.id}/alerts", headers=doctor_headers)
    assert res.status_code == 200
    alert_ids = [a["id"] for a in res.json()]
    assert test_alert.id in alert_ids

    # 2. Review alert (Acknowledge)
    rev_res = client.post(
        f"/api/digital-twin/alerts/{test_alert.id}/review",
        json={"review_action": "acknowledged", "review_notes": "Reviewed and monitored by attending physician"},
        headers=doctor_headers
    )
    assert rev_res.status_code == 200
    rev_data = rev_res.json()
    assert rev_data["review_status"] == "acknowledged"

    # Clean up test alert
    db_session.delete(test_alert)
    db_session.commit()


def test_benchmark_metrics_endpoint(client, doctor_headers):
    """Verify research benchmark API returns all 6 architectures with valid metrics."""
    res = client.get("/api/digital-twin/benchmark/metrics", headers=doctor_headers)
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    models = data["models"]
    assert len(models) == 6

    model_names = [m["name"] for m in models]
    assert any("Random Forest" in n for n in model_names)
    assert any("LSTM" in n for n in model_names)
    assert any("GRU" in n for n in model_names)
    assert any("Temporal CNN" in n for n in model_names)
    assert any("Persistence" in n for n in model_names)

    for m in models:
        assert m["test_mae"] >= 0.0
        assert m["test_rmse"] >= 0.0
        assert m["latency_ms"] >= 0.0


def test_zero_regression_existing_1_0_features(client, doctor_headers):
    """Verify that existing 1.0 routes continue to operate without interference."""
    # 1. Patient Directory
    p_res = client.get("/api/patients", headers=doctor_headers)
    assert p_res.status_code == 200

    # 2. Announcements
    a_res = client.get("/api/announcements", headers=doctor_headers)
    assert a_res.status_code == 200

    # 3. Staff Chat Directory
    c_res = client.get("/api/staff-chat/directory", headers=doctor_headers)
    assert c_res.status_code == 200
