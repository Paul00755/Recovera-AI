import uuid
import datetime as dt
from typing import Optional, Dict, Any, List
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float, Boolean, Text, Date, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class DigitalTwinState(Base):
    """
    Recovera AI 2.0: Patient Recovery Digital Twin State Entity
    Evolving deterministic digital representation of an individual patient's
    post-discharge recovery trajectory, plan compliance, and model projections.
    """
    __tablename__ = "digital_twin_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), unique=True, index=True)
    plan_version: Mapped[int] = mapped_column(Integer, default=1)
    observation_cutoff: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    
    # 5-Dimension Explainable Deviation Breakdown: D_t = w1*S + w2*A + w3*F + w4*T + w5*Q
    symptom_deviation_score: Mapped[float] = mapped_column(Float, default=0.0)      # S_t: symptom divergence
    adherence_deviation_score: Mapped[float] = mapped_column(Float, default=0.0)    # A_t: medication intake deviation
    followup_deviation_score: Mapped[float] = mapped_column(Float, default=0.0)     # F_t: appointments/vitals gap
    trajectory_slope_score: Mapped[float] = mapped_column(Float, default=0.0)       # T_t: negative trend slope (e.g. rising pain)
    uncertainty_score: Mapped[float] = mapped_column(Float, default=0.0)            # Q_t: data missingness / irregular gaps
    composite_deviation: Mapped[float] = mapped_column(Float, default=0.0)          # D_t: combined score (0.0 - 100.0)
    deviation_tier: Mapped[str] = mapped_column(String(30), default="normal")       # normal, mild, moderate, critical

    current_recovery_score: Mapped[float] = mapped_column(Float, default=100.0)    # 0 to 100 overall recovery index
    
    # Trajectory payloads for interactive visualization
    expected_trajectory_json: Mapped[Optional[Any]] = mapped_column(JSON, default=list, nullable=True)
    observed_trajectory_json: Mapped[Optional[Any]] = mapped_column(JSON, default=list, nullable=True)
    model_forecast_json: Mapped[Optional[Any]] = mapped_column(JSON, default=dict, nullable=True)
    
    updater_version: Mapped[str] = mapped_column(String(30), default="v2.0-deterministic")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


class DigitalTwinStateHistory(Base):
    """
    Auditable versioned state history recording every state transition.
    """
    __tablename__ = "digital_twin_state_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(60)) # checkin_received, med_adherence, plan_update, override
    source_event_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    plan_version: Mapped[int] = mapped_column(Integer, default=1)
    composite_deviation: Mapped[float] = mapped_column(Float, default=0.0)
    state_snapshot_json: Mapped[Optional[Any]] = mapped_column(JSON, default=dict, nullable=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)


class RecoveryTrajectoryMilestone(Base):
    """
    Clinician-approved expected numerical milestones across recovery days.
    """
    __tablename__ = "recovery_trajectory_milestones"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    day_number: Mapped[int] = mapped_column(Integer) # Day 1, 2, ..., 14
    target_date: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    expected_pain_min: Mapped[float] = mapped_column(Float, default=0.0)
    expected_pain_max: Mapped[float] = mapped_column(Float, default=10.0)
    expected_activity: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    clinical_milestone_note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class DigitalTwinPrediction(Base):
    """
    Candidate deep learning & classical model forecasts with uncertainty boundaries.
    """
    __tablename__ = "digital_twin_predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(60)) # persistence, random_forest, gru, lstm, temporal_cnn
    model_version: Mapped[str] = mapped_column(String(30), default="1.0")
    prediction_target: Mapped[str] = mapped_column(String(60), default="next_day_pain")
    prediction_horizon_hours: Mapped[int] = mapped_column(Integer, default=24)
    input_window_days: Mapped[int] = mapped_column(Integer, default=5)
    predicted_value: Mapped[float] = mapped_column(Float)
    uncertainty_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 95% CI lower
    uncertainty_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 95% CI upper
    features_used_json: Mapped[Optional[Any]] = mapped_column(JSON, default=dict, nullable=True)
    limitations_note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)


class DigitalTwinDeviationAlert(Base):
    """
    Evidence-linked deviation flags requiring clinician review.
    """
    __tablename__ = "digital_twin_deviation_alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    severity: Mapped[str] = mapped_column(String(30), default="moderate") # mild, moderate, severe
    trigger_reason: Mapped[str] = mapped_column(Text)
    trigger_evidence_json: Mapped[Optional[Any]] = mapped_column(JSON, default=dict, nullable=True)
    rule_or_model_version: Mapped[str] = mapped_column(String(60), default="D_t-rule-v2.0")
    status: Mapped[str] = mapped_column(String(30), default="pending") # pending, acknowledged, dismissed, escalated
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    clinician_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
