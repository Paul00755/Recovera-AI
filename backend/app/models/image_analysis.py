import uuid
import datetime as dt
from typing import Optional, Dict, Any, List
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float, Boolean, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class MedicalImage(Base):
    """
    Stored medical image: either a surgical wound photograph or a bone fracture radiograph (X-ray).
    """
    __tablename__ = "medical_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    uploaded_by: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    image_type: Mapped[str] = mapped_column(String(40), index=True)  # surgical_wound, bone_fracture_xray
    file_path: Mapped[str] = mapped_column(String(255))
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(50))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    anatomical_site: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Sternotomy, Abdominal, Right Tibia, Left Radius, etc.
    clinical_view: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # AP, Lateral, Oblique, Close-up, Tangential
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    captured_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    # Relationships
    patient = relationship("Patient", lazy="joined")
    uploader = relationship("User", foreign_keys=[uploaded_by], lazy="joined")
    analyses = relationship("ImageAnalysis", back_populates="image", cascade="all, delete-orphan", lazy="selectin")


class ImageAnalysisSession(Base):
    """
    Interactive analysis session tracking Draft vs Saved vs Discarded state.
    Provides non-destructive workflow with full reproducibility and audit trail.
    """
    __tablename__ = "image_analysis_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    initiated_by: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    analysis_type: Mapped[str] = mapped_column(String(40), index=True)  # surgical_wound, bone_fracture_xray
    status: Mapped[str] = mapped_column(String(30), default="draft", index=True)  # draft, saved, discarded, re_analysed
    active_analysis_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    saved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    saved_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    discarded_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    discard_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    # Relationships
    patient = relationship("Patient", lazy="joined")
    initiator = relationship("User", foreign_keys=[initiated_by], lazy="joined")
    saver = relationship("User", foreign_keys=[saved_by], lazy="joined")
    analyses = relationship("ImageAnalysis", back_populates="session", cascade="all, delete-orphan", lazy="selectin")


class ImageAnalysis(Base):
    """
    Deep learning analysis execution result with Grad-CAM and segmentation attribution.
    """
    __tablename__ = "image_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey("image_analysis_sessions.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("medical_images.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    analysis_type: Mapped[str] = mapped_column(String(40), index=True)  # surgical_wound, bone_fracture_xray
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    # Model provenance
    model_name: Mapped[str] = mapped_column(String(80))  # e.g. ResNet50-WoundNet, DenseNet121-FractureNet, U-Net++
    model_version: Mapped[str] = mapped_column(String(40), default="v1.0-prod")
    model_architecture: Mapped[str] = mapped_column(String(80))
    weights_provenance: Mapped[str] = mapped_column(String(120))  # e.g. ImageNet1K + FineTuned-SurgWound

    # Output predictions & probabilities
    primary_prediction: Mapped[str] = mapped_column(String(80))  # e.g. Normal Healing, Erythematous/Complicated, Fracture Detected, No Fracture
    confidence_score: Mapped[float] = mapped_column(Float)  # 0.0 - 1.0
    probabilities_json: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)
    findings_summary: Mapped[str] = mapped_column(Text)
    severity_level: Mapped[str] = mapped_column(String(30), default="low")  # low, moderate, elevated, critical

    # Explainability artifacts
    gradcam_heatmap_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    gradcam_overlay_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    segmentation_mask_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    segmentation_overlay_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metrics_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)

    # Clinician Review
    review_status: Mapped[str] = mapped_column(String(30), default="pending_review", index=True)  # pending_review, approved, rejected, amended
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    clinician_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clinician_agreed_with_ai: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    doctor_confirmed_findings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_patient_visible: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    # Relationships
    session = relationship("ImageAnalysisSession", back_populates="analyses")
    image = relationship("MedicalImage", back_populates="analyses")
    patient = relationship("Patient", lazy="joined")
    reviewer = relationship("User", foreign_keys=[reviewed_by], lazy="joined")
    wound_observation = relationship("WoundObservation", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    fracture_analysis = relationship("FractureAnalysis", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    gemini_reassessment = relationship("GeminiReassessment", back_populates="analysis", uselist=False, cascade="all, delete-orphan")


class WoundObservation(Base):
    """
    Granular clinical measurements extracted from surgical wound photograph and segmentation.
    """
    __tablename__ = "wound_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("image_analyses.id", ondelete="CASCADE"), unique=True, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)

    # Geometric & Segmentation measurements
    wound_area_pixels: Mapped[int] = mapped_column(Integer, default=0)
    wound_area_cm2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wound_perimeter_pixels: Mapped[int] = mapped_column(Integer, default=0)
    relative_wound_size_pct: Mapped[float] = mapped_column(Float, default=0.0)
    closure_rate_cm2_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Visible clinical characteristics
    erythema_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 - 100.0 redness index
    edema_present: Mapped[bool] = mapped_column(Boolean, default=False)
    dehiscence_risk_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 - 100.0 margin separation
    exudate_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    exudate_type: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)  # serous, purulent, sanguineous, none
    predominant_tissue_type: Mapped[str] = mapped_column(String(50), default="granulation")  # granulation, slough, necrotic, epithelializing
    healing_trajectory_status: Mapped[str] = mapped_column(String(40), default="on_track")  # on_track, delayed, stagnant, deteriorating

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    analysis = relationship("ImageAnalysis", back_populates="wound_observation")


class FractureAnalysis(Base):
    """
    Granular radiographic findings extracted from bone fracture X-ray.
    """
    __tablename__ = "fracture_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("image_analyses.id", ondelete="CASCADE"), unique=True, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)

    fracture_detected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fracture_probability: Mapped[float] = mapped_column(Float, default=0.0)
    anatomical_bone: Mapped[str] = mapped_column(String(80), default="Unspecified Bone")  # Tibia, Fibula, Radius, Femur, Clavicle, etc.
    fracture_type: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)  # transverse, oblique, spiral, comminuted, hairline, none
    displacement_present: Mapped[bool] = mapped_column(Boolean, default=False)
    displacement_severity: Mapped[str] = mapped_column(String(30), default="none")  # none, minimal, moderate, severe
    callus_formation_observed: Mapped[bool] = mapped_column(Boolean, default=False)
    healing_stage: Mapped[str] = mapped_column(String(40), default="acute")  # acute, early_callus, consolidated, remodeled
    localization_box_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    analysis = relationship("ImageAnalysis", back_populates="fracture_analysis")


class GeminiReassessment(Base):
    """
    Clinical reassessment generated by Google Gemini API after an image analysis is saved.
    Cross-evaluates medical history, active prescriptions, check-in telemetry, and new image findings.
    """
    __tablename__ = "gemini_reassessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("image_analyses.id", ondelete="CASCADE"), unique=True, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(60), default="gemini-3.5-flash-lite")
    model_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # Synthesis outputs
    reassessment_summary: Mapped[str] = mapped_column(Text)
    relevant_new_findings: Mapped[List[str]] = mapped_column(JSON, default=list)
    changes_vs_previous: Mapped[str] = mapped_column(Text)
    potential_concerns: Mapped[List[str]] = mapped_column(JSON, default=list)
    suggested_clinician_actions: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Governance recommendations (requires MD approval)
    recovery_plan_review_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    recovery_plan_proposed_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_review_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    recommended_risk_tier: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # green, yellow, red
    risk_escalation_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    uncertainties_and_missing_data: Mapped[List[str]] = mapped_column(JSON, default=list)
    source_records_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    processing_status: Mapped[str] = mapped_column(String(30), default="completed")  # completed, fallback_rule_based, failed
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    # Relationships
    analysis = relationship("ImageAnalysis", back_populates="gemini_reassessment")


class AnalysisExport(Base):
    """
    Audit log of clinical report exports (PDF, CSV, image bundles).
    """
    __tablename__ = "analysis_exports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id: Mapped[Optional[str]] = mapped_column(ForeignKey("image_analyses.id", ondelete="SET NULL"), nullable=True, index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    exported_by: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    export_format: Mapped[str] = mapped_column(String(20))  # pdf, csv, image_bundle
    file_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    includes_gemini_reassessment: Mapped[bool] = mapped_column(Boolean, default=True)
    includes_gradcam: Mapped[bool] = mapped_column(Boolean, default=True)
    includes_segmentation: Mapped[bool] = mapped_column(Boolean, default=True)
    exported_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    # Relationships
    exporter = relationship("User", foreign_keys=[exported_by], lazy="joined")
    patient = relationship("Patient", lazy="joined")


class ModelTrainingRun(Base):
    """
    Research benchmark registry recording reproducible transfer-learning runs, datasets, hyperparameters, and evaluation metrics.
    """
    __tablename__ = "model_training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_type: Mapped[str] = mapped_column(String(40), index=True)  # surgical_wound, bone_fracture
    architecture: Mapped[str] = mapped_column(String(60))  # ResNet50, DenseNet121, EfficientNet-B3, U-Net++
    pretrained_weights: Mapped[str] = mapped_column(String(60), default="ImageNet1K")
    dataset_name: Mapped[str] = mapped_column(String(60))  # SurgWound, FracAtlas, MURA, FUSeg
    dataset_version: Mapped[str] = mapped_column(String(30), default="v1.0")
    total_samples: Mapped[int] = mapped_column(Integer)
    train_samples: Mapped[int] = mapped_column(Integer)
    val_samples: Mapped[int] = mapped_column(Integer)
    test_samples: Mapped[int] = mapped_column(Integer)
    patient_level_split: Mapped[bool] = mapped_column(Boolean, default=True)  # prevents data leakage

    # Hyperparameters
    epochs: Mapped[int] = mapped_column(Integer)
    batch_size: Mapped[int] = mapped_column(Integer)
    learning_rate: Mapped[float] = mapped_column(Float)
    optimizer: Mapped[str] = mapped_column(String(30), default="AdamW")
    loss_function: Mapped[str] = mapped_column(String(60))

    # Evaluation metrics
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0)
    roc_auc: Mapped[float] = mapped_column(Float, default=0.0)
    pr_auc: Mapped[float] = mapped_column(Float, default=0.0)
    sensitivity: Mapped[float] = mapped_column(Float, default=0.0)
    specificity: Mapped[float] = mapped_column(Float, default=0.0)
    dice_coefficient: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # for segmentation
    iou_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # for segmentation
    confusion_matrix_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    inference_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    training_date: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    version_tag: Mapped[str] = mapped_column(String(40), default="v1.0")
    is_production_model: Mapped[bool] = mapped_column(Boolean, default=False)
