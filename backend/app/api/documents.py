import os
import uuid
import datetime as dt
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_db
from app.core.config import settings
from app.core.constants import Role, DocStatus
from app.core.audit import record_audit
from app.models.clinical_document import ClinicalDocument, OCRExtraction
from app.models.patient import Patient
from app.models.user import User
from app.schemas.document import DocumentReviewPatch
from app.services.ocr_service import extract_text_from_file, clean_ocr_text, parse_prescription_fields
from app.services.recovery_service import invalidate_plan_on_source_change
from app.api.deps import get_current_user, require_role, enforce_patient_isolation

router = APIRouter(tags=["Clinical Documents & PaddleOCR"])

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

@router.post("/api/patients/{pid}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    pid: str,
    doc_type: str = Form(..., pattern="^(prescription|discharge_summary|recovery_instructions|follow_up_schedule)$"),
    file: UploadFile = File(...),
    target_doctor_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Determine designated approver doctor
    assigned_doc_id = None
    if current_user.role == Role.DOCTOR.value:
        assigned_doc_id = current_user.id
    elif target_doctor_id:
        assigned_doc_id = target_doctor_id
    elif patient.assigned_doctor_id:
        assigned_doc_id = patient.assigned_doctor_id

    # Save file safely to storage directory
    file_id = str(uuid.uuid4())
    stored_filename = f"{patient.patient_code}_{file_id}{ext}"
    patient_dir = Path(settings.UPLOAD_DIR) / patient.id
    patient_dir.mkdir(parents=True, exist_ok=True)
    destination_path = patient_dir / stored_filename

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds the 15MB limit.")

    with open(destination_path, "wb") as f:
        f.write(contents)

    # Run ML Handwriting & OCR Text Extraction
    from app.services.handwriting_ml_service import HandwrittenMLService
    ml_res = HandwrittenMLService.process_handwritten_photo(str(destination_path))
    raw_text = ml_res["raw_text"] or extract_text_from_file(str(destination_path))
    cleaned = ml_res["clean_text"] or clean_ocr_text(raw_text)

    # Document-aware extraction with ML prescription parsing
    structured_data = {}
    if doc_type == "prescription":
        meds = ml_res.get("medications") or parse_prescription_fields(cleaned)
        structured_data = {
            "medications": meds,
            "is_handwritten_photo": ml_res.get("is_photo", False),
            "ml_model_version": ml_res.get("model_version"),
            "preprocessing": ml_res.get("preprocessing", {})
        }
    elif doc_type == "follow_up_schedule":
        structured_data = {"raw_schedule": cleaned}
    elif doc_type in ("discharge_summary", "recovery_instructions"):
        structured_data = {"instructions": cleaned}

    # Create ClinicalDocument record
    doc = ClinicalDocument(
        id=file_id,
        patient_id=patient.id,
        doc_type=doc_type,
        filename=file.filename or f"doc_{file_id}{ext}",
        stored_path=str(destination_path),
        status=DocStatus.DRAFT.value,
        version=1,
        extracted_text=cleaned,
        structured_data=structured_data,
        uploaded_by=current_user.id,
        assigned_doctor_id=assigned_doc_id,
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow()
    )
    db.add(doc)
    db.flush()

    # Save OCRExtraction metadata
    extraction = OCRExtraction(
        document_id=doc.id,
        raw_text=raw_text,
        cleaned_text=cleaned,
        extracted_fields=structured_data,
        confidence_scores={"overall": 0.90 if cleaned else 0.0},
        created_at=dt.datetime.utcnow()
    )
    db.add(extraction)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="document_uploaded_ocr",
        target_type="clinical_document",
        target_id=doc.id,
        new_values={"filename": doc.filename, "doc_type": doc_type, "status": doc.status, "assigned_doctor_id": assigned_doc_id}
    )

    # Notify designated doctor if uploaded by nurse
    if assigned_doc_id and current_user.role == Role.NURSE.value:
        from app.services.notification_service import create_in_app_notification
        pat_name = patient.user.name if patient.user else patient.patient_code
        create_in_app_notification(
            db=db,
            user_id=assigned_doc_id,
            title="Document Pending Review",
            message=f"Nurse {current_user.name} uploaded a {doc_type.replace('_', ' ')} for patient {pat_name} awaiting your approval.",
            category="document",
            link=f"#/staff/patients/{patient.id}/documents"
        )

    db.commit()

    return {
        "id": doc.id,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "status": doc.status,
        "extracted_text": doc.extracted_text,
        "structured_data": doc.structured_data,
        "assigned_doctor_id": doc.assigned_doctor_id,
        "created_at": doc.created_at.isoformat()
    }

@router.get("/api/patients/{pid}/documents")
def list_patient_documents(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from sqlalchemy import or_
    patient = enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)
    query = select(ClinicalDocument).where(ClinicalDocument.patient_id == patient.id).order_by(ClinicalDocument.created_at.desc())

    # Patient role can ONLY view APPROVED documents
    if current_user.role == Role.PATIENT.value:
        query = query.where(ClinicalDocument.status == DocStatus.APPROVED.value)
    elif current_user.role == Role.DOCTOR.value:
        from app.models.patient import PatientDoctorAssignment
        is_assigned_to_patient = (patient.assigned_doctor_id == current_user.id) or bool(
            db.scalar(
                select(PatientDoctorAssignment.id).where(
                    PatientDoctorAssignment.patient_id == patient.id,
                    PatientDoctorAssignment.doctor_id == current_user.id,
                    PatientDoctorAssignment.is_active == True
                )
            )
        )
        if is_assigned_to_patient:
            query = query.where(
                or_(
                    ClinicalDocument.uploaded_by == current_user.id,
                    ClinicalDocument.assigned_doctor_id == current_user.id,
                    ClinicalDocument.status == DocStatus.APPROVED.value
                )
            )
        else:
            query = query.where(
                or_(
                    ClinicalDocument.uploaded_by == current_user.id,
                    ClinicalDocument.assigned_doctor_id == current_user.id
                )
            )

    docs = db.scalars(query).all()

    # Pre-fetch users for names
    u_ids = list({d.uploaded_by for d in docs}.union({d.assigned_doctor_id for d in docs if d.assigned_doctor_id}))
    user_map = {}
    if u_ids:
        users = db.scalars(select(User).where(User.id.in_(u_ids))).all()
        user_map = {u.id: u for u in users}

    return [
        {
            "id": d.id,
            "patient_id": d.patient_id,
            "doc_type": d.doc_type,
            "filename": d.filename,
            "status": d.status,
            "version": d.version,
            "extracted_text": d.extracted_text,
            "structured_data": d.structured_data,
            "uploaded_by": d.uploaded_by,
            "uploader_name": user_map.get(d.uploaded_by).name if user_map.get(d.uploaded_by) else "Staff",
            "uploader_role": user_map.get(d.uploaded_by).role if user_map.get(d.uploaded_by) else "staff",
            "assigned_doctor_id": d.assigned_doctor_id,
            "assigned_doctor_name": user_map.get(d.assigned_doctor_id).name if user_map.get(d.assigned_doctor_id) else None,
            "can_approve": (current_user.role in (Role.ADMIN.value, Role.NURSE.value)) or (current_user.role == Role.DOCTOR.value and (d.uploaded_by == current_user.id or d.assigned_doctor_id == current_user.id or is_assigned_to_patient)),
            "reviewed_by": d.reviewed_by,
            "created_at": d.created_at.isoformat(),
            "updated_at": d.updated_at.isoformat()
        }
        for d in docs
    ]

@router.get("/api/documents/{doc_id}")
def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Isolation check
    enforce_patient_isolation(patient_id=doc.patient_id, current_user=current_user, db=db)

    # If patient, ensure document is approved
    if current_user.role == Role.PATIENT.value and doc.status != DocStatus.APPROVED.value:
        raise HTTPException(status_code=403, detail="Unapproved documents are not accessible to patients.")

    # If doctor, must be uploader or assigned doctor
    if current_user.role == Role.DOCTOR.value and (doc.uploaded_by != current_user.id and doc.assigned_doctor_id != current_user.id):
        raise HTTPException(status_code=403, detail="Access denied: This clinical document is restricted to its designated clinician.")

    uploader = db.get(User, doc.uploaded_by)
    assigned_doc = db.get(User, doc.assigned_doctor_id) if doc.assigned_doctor_id else None

    # Check doctor patient assignment
    is_assigned = False
    if current_user.role == Role.DOCTOR.value:
        from app.models.patient import PatientDoctorAssignment
        is_assigned = (doc.assigned_doctor_id == current_user.id) or (doc.uploaded_by == current_user.id) or bool(
            db.scalar(
                select(PatientDoctorAssignment.id).where(
                    PatientDoctorAssignment.patient_id == doc.patient_id,
                    PatientDoctorAssignment.doctor_id == current_user.id,
                    PatientDoctorAssignment.is_active == True
                )
            )
        )

    return {
        "id": doc.id,
        "patient_id": doc.patient_id,
        "doc_type": doc.doc_type,
        "filename": doc.filename,
        "status": doc.status,
        "version": doc.version,
        "extracted_text": doc.extracted_text,
        "structured_data": doc.structured_data,
        "uploaded_by": doc.uploaded_by,
        "uploader_name": uploader.name if uploader else "Staff",
        "assigned_doctor_id": doc.assigned_doctor_id,
        "assigned_doctor_name": assigned_doc.name if assigned_doc else None,
        "can_approve": (current_user.role in (Role.ADMIN.value, Role.NURSE.value)) or is_assigned,
        "reviewed_by": doc.reviewed_by,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat()
    }

@router.patch("/api/documents/{doc_id}")
def review_and_edit_document(
    doc_id: str,
    req: DocumentReviewPatch,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Doctor check: only uploader or assigned doctor
    if current_user.role == Role.DOCTOR.value and (doc.uploaded_by != current_user.id and doc.assigned_doctor_id != current_user.id):
        raise HTTPException(status_code=403, detail="Access denied: You are not authorized to modify this document.")

    old_status = doc.status
    if req.extracted_text is not None:
        doc.extracted_text = req.extracted_text
    if req.structured_data is not None:
        doc.structured_data = req.structured_data
    if req.assigned_doctor_id:
        doc.assigned_doctor_id = req.assigned_doctor_id
    if req.status:
        doc.status = req.status
    else:
        doc.status = DocStatus.UNDER_REVIEW.value

    doc.reviewed_by = current_user.id
    doc.updated_at = dt.datetime.utcnow()

    # If nurse submits for approval, notify the assigned doctor
    if doc.status == DocStatus.UNDER_REVIEW.value and doc.assigned_doctor_id and current_user.role == Role.NURSE.value:
        from app.services.notification_service import create_in_app_notification
        pat = db.get(Patient, doc.patient_id)
        pat_name = pat.user.name if (pat and pat.user) else (pat.patient_code if pat else "Patient")
        create_in_app_notification(
            db=db,
            user_id=doc.assigned_doctor_id,
            title="Document Submitted for Approval",
            message=f"Nurse {current_user.name} reviewed and submitted {doc.doc_type.replace('_', ' ')} for patient {pat_name} for your approval.",
            category="document",
            link=f"#/staff/patients/{doc.patient_id}/documents"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="document_review_edited",
        target_type="clinical_document",
        target_id=doc.id,
        old_values={"status": old_status},
        new_values={"status": doc.status}
    )
    db.commit()

    return {"message": "Document updated.", "status": doc.status, "id": doc.id}

@router.post("/api/documents/{doc_id}/approve")
def approve_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Doctor-specific approval restriction:
    # Only the doctor who uploaded the document, was designated for its approval, or is assigned to the patient can approve it
    if current_user.role == Role.DOCTOR.value:
        from app.models.patient import PatientDoctorAssignment
        is_assigned_care = (doc.assigned_doctor_id == current_user.id) or bool(
            db.scalar(
                select(PatientDoctorAssignment.id).where(
                    PatientDoctorAssignment.patient_id == doc.patient_id,
                    PatientDoctorAssignment.doctor_id == current_user.id,
                    PatientDoctorAssignment.is_active == True
                )
            )
        )
        patient_rec = db.get(Patient, doc.patient_id)
        if patient_rec and patient_rec.assigned_doctor_id == current_user.id:
            is_assigned_care = True

        if doc.uploaded_by != current_user.id and not is_assigned_care:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Only the clinician who uploaded, was designated for this document, or is assigned to the patient can approve it."
            )

    doc.status = DocStatus.APPROVED.value
    doc.reviewed_by = current_user.id
    doc.updated_at = dt.datetime.utcnow()

    # 1. Ingest approved prescription medications into patient's active Medication and schedule tables
    ingested_meds_count = 0
    if doc.doc_type == "prescription" and doc.structured_data:
        from app.models.medication import Medication, MedicationSchedule
        from app.services.medication_service import generate_occurrences_for_schedules
        meds_list = doc.structured_data.get("medications", [])
        today = dt.date.today()

        for med in meds_list:
            m_name = med.get("name") or med.get("medicine_name")
            if not m_name:
                continue

            days_count = int(med.get("days", 5)) if str(med.get("days", "")).isdigit() else 5
            start_date = today
            end_date = today + dt.timedelta(days=max(0, days_count - 1))
            dose_val = med.get("dose") or med.get("dosage") or "1 tablet"
            food_rel = med.get("food_relation") or "after_food"
            period_label = med.get("period_label") or med.get("period") or "Daily"
            instructions = med.get("instructions") or f"Take {dose_val} ({period_label}) for {days_count} days."

            # Automatically register extracted medicine into system-wide medicine library if new
            try:
                from app.services.handwriting_ml_service import register_medicine_in_library
                register_medicine_in_library(
                    name=m_name,
                    generic=med.get("generic_name") or med.get("generic"),
                    typical_doses=[dose_val] if dose_val and dose_val != "1 tablet" else None,
                    forms=[med.get("form", "Tablet")]
                )
            except Exception as lib_err:
                logger.warning(f"Failed to register medicine '{m_name}' in library: {lib_err}")

            # Check if active medication already exists
            existing_med = db.scalar(
                select(Medication).where(
                    Medication.patient_id == doc.patient_id,
                    Medication.name.ilike(m_name),
                    Medication.status == "active"
                )
            )
            if existing_med:
                existing_med.end_date = max(existing_med.end_date, end_date)
                existing_med.instructions = instructions
                existing_med.version = (existing_med.version or 1) + 1
                med_obj = existing_med
            else:
                med_obj = Medication(
                    patient_id=doc.patient_id,
                    name=m_name.title(),
                    generic_name=med.get("generic_name") or m_name.title(),
                    strength=med.get("strength") or dose_val,
                    dose=dose_val,
                    unit="oral",
                    route="oral",
                    start_date=start_date,
                    end_date=end_date,
                    instructions=instructions,
                    status="active",
                    version=1,
                    source_document_id=doc.id,
                    created_by=current_user.id,
                    created_at=dt.datetime.utcnow()
                )
                db.add(med_obj)
                db.flush()

                created_scheds = []
                sched_list = med.get("schedules") or []
                if sched_list:
                    for s in sched_list:
                        sched = MedicationSchedule(
                            medication_id=med_obj.id,
                            patient_id=doc.patient_id,
                            label=s.get("label", "Dose"),
                            time_hhmm=s.get("time_hhmm", "08:00"),
                            dose=s.get("dose", dose_val),
                            food_relation=s.get("food_relation", food_rel)
                        )
                        db.add(sched)
                        created_scheds.append(sched)
                else:
                    default_sched = MedicationSchedule(
                        medication_id=med_obj.id,
                        patient_id=doc.patient_id,
                        label="Morning",
                        time_hhmm="08:00",
                        dose=dose_val,
                        food_relation=food_rel
                    )
                    db.add(default_sched)
                    created_scheds.append(default_sched)

                db.flush()
                generate_occurrences_for_schedules(
                    medication=med_obj,
                    schedules=created_scheds,
                    start_date=start_date,
                    end_date=end_date,
                    db=db
                )
                ingested_meds_count += 1

    # 2. Automatically generate / refresh patient's evidence-grounded Recovery Plan
    try:
        from app.services.recovery_service import generate_plan_from_approved_documents
        generate_plan_from_approved_documents(patient_id=doc.patient_id, db=db, user_id=current_user.id)
    except Exception:
        pass

    # Suggestion 7: Notify uploading nurse and patient of approved document
    try:
        from app.services.notification_service import notify_document_approved
        notify_document_approved(db=db, doc=doc, doctor=current_user)
    except Exception as e:
        import logging
        logging.getLogger("recovera.documents").warning("Failed to notify document approval: %s", e)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="document_approved",
        target_type="clinical_document",
        target_id=doc.id,
        new_values={
            "status": DocStatus.APPROVED.value,
            "ingested_medications": ingested_meds_count
        }
    )
    db.commit()

    return {
        "message": "Document approved. Prescriptions ingested into Medications tracker and care plan updated.",
        "id": doc.id,
        "status": doc.status,
        "ingested_medications": ingested_meds_count
    }

@router.post("/api/documents/{doc_id}/reject")
def reject_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Doctor-specific approval/rejection restriction
    if current_user.role == Role.DOCTOR.value:
        if doc.uploaded_by != current_user.id and doc.assigned_doctor_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Only the clinician who uploaded or was designated for this document can reject it."
            )

    doc.status = DocStatus.REJECTED.value
    doc.reviewed_by = current_user.id
    doc.updated_at = dt.datetime.utcnow()

    invalidate_plan_on_source_change(patient_id=doc.patient_id, db=db, document_id=doc.id)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="document_rejected",
        target_type="clinical_document",
        target_id=doc.id,
        new_values={"status": DocStatus.REJECTED.value}
    )
    db.commit()

    return {"message": "Document rejected. It will not be visible to patients.", "id": doc.id, "status": doc.status}

@router.delete("/api/documents/{doc_id}")
def supersede_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.status = DocStatus.SUPERSEDED.value
    doc.updated_at = dt.datetime.utcnow()

    invalidate_plan_on_source_change(patient_id=doc.patient_id, db=db, document_id=doc.id)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="document_superseded",
        target_type="clinical_document",
        target_id=doc.id,
        new_values={"status": DocStatus.SUPERSEDED.value}
    )
    db.commit()

    return {"message": "Document marked as superseded.", "id": doc.id, "status": doc.status}

@router.get("/api/documents/{doc_id}/download")
def download_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Isolation check
    enforce_patient_isolation(patient_id=doc.patient_id, current_user=current_user, db=db)

    if current_user.role == Role.PATIENT.value and doc.status != DocStatus.APPROVED.value:
        raise HTTPException(status_code=403, detail="Access denied.")

    stored_file = doc.stored_path
    if not os.path.exists(stored_file):
        fallback_candidate = Path(settings.UPLOAD_DIR) / doc.patient_id / Path(doc.stored_path).name
        if fallback_candidate.exists():
            stored_file = str(fallback_candidate)
        else:
            raise HTTPException(status_code=404, detail="Physical document file missing on server.")

    return FileResponse(
        path=stored_file,
        filename=doc.filename,
        media_type="application/octet-stream"
    )

@router.post("/api/documents/{doc_id}/re-extract-ml")
def re_extract_handwriting_ml(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    doc = db.get(ClinicalDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    stored_file = doc.stored_path
    if not os.path.exists(stored_file):
        fallback_candidate = Path(settings.UPLOAD_DIR) / doc.patient_id / Path(doc.stored_path).name
        if fallback_candidate.exists():
            stored_file = str(fallback_candidate)
        else:
            raise HTTPException(status_code=404, detail="Document file missing on disk.")

    from app.services.handwriting_ml_service import HandwrittenMLService
    ml_res = HandwrittenMLService.process_handwritten_photo(stored_file)

    if ml_res.get("clean_text"):
        doc.extracted_text = ml_res["clean_text"]
    if doc.doc_type == "prescription":
        doc.structured_data = {
            "medications": ml_res.get("medications", []),
            "is_handwritten_photo": ml_res.get("is_photo", False),
            "ml_model_version": ml_res.get("model_version"),
            "preprocessing": ml_res.get("preprocessing", {})
        }
    doc.updated_at = dt.datetime.utcnow()
    db.commit()

    return {
        "message": "Handwriting ML re-extraction completed.",
        "extracted_text": doc.extracted_text,
        "structured_data": doc.structured_data,
        "medications": ml_res.get("medications", [])
    }

