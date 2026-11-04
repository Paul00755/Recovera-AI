"""
Recovera AI – Grounded Universal Recovery Assistant API (Phase 11 & Gemini Upgrade)

Provides:
  GET    /api/recovery-assistant/sessions       – List chat sessions for current user (left sidebar)
  POST   /api/recovery-assistant/sessions       – Create new chat session (+ New Chat)
  GET    /api/recovery-assistant/sessions/{id}  – Get message history of a chat session
  DELETE /api/recovery-assistant/sessions/{id}  – Delete a chat session
  POST   /api/recovery-assistant/chat           – Chat with Gemini 2.5 Flash / Bulletproof Offline Fallback
  GET    /api/recovery-assistant/context        – Grounding context telemetry
"""
import uuid
import datetime as dt
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.constants import Role
from app.models.user import User
from app.models.patient import Patient
from app.models.chat_session import ChatSession, ChatMessage
from app.api.deps import get_current_user
from app.services.recovery_ai_service import (
    gather_clinical_context,
    call_gemini_assistant,
    analyze_uploaded_file,
    is_healthcare_relevant,
    get_gemini_api_key,
    translate_clinical_texts,
)

router = APIRouter(tags=["Recovery Assistant"])


class FileAttachmentPayload(BaseModel):
    filename: str
    mime_type: str
    data_base64: str  # Base64 encoded payload


class RecoveryChatRequest(BaseModel):
    session_id: Optional[str] = None
    patient_id: Optional[str] = None
    message: Optional[str] = None
    query: Optional[str] = None
    language: str = "en"  # "en", "ta", "hi"
    attachment: Optional[FileAttachmentPayload] = None


class RecoveryChatResponse(BaseModel):
    session_id: str
    reply: str
    answer: str  # Duplicate alias for backward compatibility with frontend
    is_emergency: bool = False
    emergency_escalation: bool = False
    prescription_boundary_enforced: bool = False
    grounded_source: str = "approved_recovery_plan"
    plan_found: bool = True
    suggested_actions: List[str] = []
    citations: List[Dict[str, Any]] = []
    attending_doctor: Optional[str] = None
    model_used: str = "Gemini 3.8 Flash"
    metadata: Dict[str, Any] = {}


class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Consultation"
    language: Optional[str] = "en"


class TranslateItemPayload(BaseModel):
    id: Optional[str] = None
    text: str


class ClinicalTranslateRequest(BaseModel):
    items: Optional[List[TranslateItemPayload]] = None
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    target_lang: str = "en"  # "en", "ta", "hi"
    source_lang: Optional[str] = None


class ClinicalTranslateResponse(BaseModel):
    status: str = "success"
    items: List[Dict[str, Any]] = []
    translations: List[str] = []
    translated_text: Optional[str] = None
    target_lang: str


class ChatMessageOut(BaseModel):
    id: str
    sender: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    attachment_info: Optional[Dict[str, Any]] = None
    is_emergency: bool = False
    prescription_boundary: bool = False
    model_used: str = "gemini-2.5-flash"
    suggested_actions: Optional[List[str]] = None
    created_at: dt.datetime


class ChatSessionOut(BaseModel):
    id: str
    role: str
    title: str
    language: str
    created_at: dt.datetime
    updated_at: dt.datetime
    message_count: int = 0


@router.get("/api/recovery-assistant/sessions", response_model=List[ChatSessionOut])
def list_chat_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lists all conversation sessions belonging to the authenticated user for the sidebar."""
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(desc(ChatSession.updated_at))
    ).all()

    out = []
    for s in sessions:
        count = len(s.messages) if s.messages else 0
        out.append(ChatSessionOut(
            id=s.id,
            role=s.role,
            title=s.title,
            language=s.language or "en",
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=count
        ))
    return out


@router.post("/api/recovery-assistant/sessions", response_model=ChatSessionOut)
def create_chat_session(
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Creates a fresh chat session for '+ New Chat'."""
    patient_id = None
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if patient:
            patient_id = patient.id

    new_session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        patient_id=patient_id,
        role=current_user.role,
        title=payload.title or "New Consultation",
        language=payload.language or "en",
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return ChatSessionOut(
        id=new_session.id,
        role=new_session.role,
        title=new_session.title,
        language=new_session.language,
        created_at=new_session.created_at,
        updated_at=new_session.updated_at,
        message_count=0
    )


@router.get("/api/recovery-assistant/sessions/{session_id}")
def get_chat_session_details(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves full message history of a given chat session."""
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    msgs = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    ).all()

    return {
        "session": {
            "id": session.id,
            "title": session.title,
            "language": session.language,
            "role": session.role,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "messages": [
            {
                "id": m.id,
                "sender": m.sender,
                "content": m.content,
                "text": m.content,  # frontend convenience
                "citations": m.citations or [],
                "attachment_info": m.attachment_info,
                "is_emergency": m.is_emergency,
                "prescription_boundary": m.prescription_boundary,
                "model_used": m.model_used,
                "suggested_actions": m.suggested_actions or [],
                "created_at": m.created_at.strftime("%I:%M %p") if m.created_at else "",
            }
            for m in msgs
        ]
    }


@router.delete("/api/recovery-assistant/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deletes a chat session and all its messages."""
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    db.delete(session)
    db.commit()
    return {"status": "success", "deleted_session_id": session_id}


@router.post("/api/recovery-assistant/chat", response_model=RecoveryChatResponse)
def recovery_chat(
    req: RecoveryChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Main conversational endpoint:
    - Resolves or creates conversation session.
    - Gathers dynamic database context (medications, schedules, food timings, purposes, check-ins, appointments).
    - Validates file attachment (rejects non-medical uploads).
    - Calls Gemini 2.5 Flash with fallback to offline clinical engine (zero user errors).
    - Persists messages and auto-titles new sessions.
    """
    user_query = (req.message or req.query or "").strip()
    has_attachment = req.attachment is not None

    if not user_query and not has_attachment:
        raise HTTPException(status_code=422, detail="Message or attachment is required.")

    # 1. Resolve Patient ID
    target_patient_id = req.patient_id
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if patient:
            target_patient_id = patient.id
    elif not target_patient_id and current_user.role != Role.ADMIN.value:
        # For doctors/nurses without a specified patient, query general caseload
        target_patient_id = None

    # 2. File Attachment Inspection & Guardrail
    file_bytes = None
    file_mime = None
    file_name = None
    file_desc = None
    attachment_record = None

    if req.attachment:
        is_med_file, file_desc_text, raw_b = analyze_uploaded_file(
            req.attachment.data_base64,
            req.attachment.mime_type,
            req.attachment.filename
        )
        if not is_med_file:
            if file_desc_text == "RESTRICTED_MEDIA_TYPE":
                refusal_msg = (
                    "⚠️ **Audio/Video Media Not Supported:**\n\n"
                    "Recovera AI cannot accept audio (.mp3, .wav, .m4a) or video (.mp4, .mkv, .avi) files. "
                    "Please upload a clinical document such as a PDF, DOCX, TXT consultation transcript, or prescription photo."
                )
                if req.language == "ta":
                    refusal_msg = (
                        "⚠️ **ஆடியோ / வீடியோ கோப்புகள் ஏற்கப்படாது:**\n\n"
                        "ரெக்கவரா AI ஆடியோ (.mp3, .wav) அல்லது வீடியோ (.mp4, .mkv) கோப்புகளை ஏற்காது. "
                        "தயவுசெய்து PDF, DOCX, TXT மருத்துவ ஆலோசனைப் பிரதி அல்லது மருந்துச்சீட்டு புகைப்படத்தை பதிவேற்றவும்."
                    )
                elif req.language == "hi":
                    refusal_msg = (
                        "⚠️ **ऑडियो/वीडियो फ़ाइलें समर्थित नहीं हैं:**\n\n"
                        "रिकवरा एआई ऑडियो (.mp3, .wav) या वीडियो (.mp4, .mkv) फ़ाइलों का समर्थन नहीं करता है। "
                        "कृपया केवल चिकित्सीय दस्तावेज़ जैसे पीडीएफ (PDF), वर्ड (DOCX), टेक्स्ट या नुस्खे की तस्वीर अपलोड करें।"
                    )
                sugg = ["Upload PDF Transcript", "Upload Prescription Image", "Upload DOCX Report"]
            else:
                # Reject non-medical file upload
                refusal_msg = (
                    "⚠️ **Non-Healthcare File Detected:**\n\n"
                    "Recovera AI only accepts medical documents, consultation transcripts, lab reports, or prescription photos. "
                    "The uploaded file does not appear to contain relevant healthcare information. Please upload a valid clinical document."
                )
                if req.language == "ta":
                    refusal_msg = (
                        "⚠️ **மருத்துவமற்ற கோப்பு கண்டறியப்பட்டது:**\n\n"
                        "ரெக்கவரா AI மருத்துவ ஆவணங்கள், மருத்துவ ஆலோசனைப் பிரதிகள், ஆய்வக அறிக்கைகள் அல்லது மருந்துச்சீட்டு புகைப்படங்களை மட்டுமே ஏற்கும். "
                        "தயவுசெய்து சரியான மருத்துவ ஆவணத்தை பதிவேற்றவும்."
                    )
                elif req.language == "hi":
                    refusal_msg = (
                        "⚠️ **गैर-चिकित्सीय फ़ाइल पाई गई:**\n\n"
                        "रिकवरा एआई केवल चिकित्सीय दस्तावेज़, परामर्श प्रतिलेख, लैब रिपोर्ट या नुस्खे की तस्वीरों का विश्लेषण करता है। "
                        "कृपया केवल वैध स्वास्थ्य दस्तावेज़ अपलोड करें।"
                    )
                sugg = ["Upload Consultation Transcript (PDF)", "Upload Prescription Photo"]
            return RecoveryChatResponse(
                session_id=req.session_id or "rejected",
                reply=refusal_msg,
                answer=refusal_msg,
                is_emergency=False,
                emergency_escalation=False,
                prescription_boundary_enforced=False,
                grounded_source="file_safety_guardrail",
                plan_found=True,
                suggested_actions=sugg,
                citations=[],
                model_used="Recovera Security Filter"
            )

        file_bytes = raw_b
        file_mime = req.attachment.mime_type
        file_name = req.attachment.filename
        file_desc = file_desc_text
        attachment_record = {
            "filename": file_name,
            "mime_type": file_mime,
            "size_bytes": len(raw_b) if raw_b else 0
        }

    # 3. Gather Real-Time Database Grounding Context
    ctx = gather_clinical_context(db, current_user, target_patient_id)

    # 4. Invoke Gemini with Silent Fallback
    query_text = user_query if user_query else "Please analyze the attached medical document."
    ai_result = call_gemini_assistant(
        user_query=query_text,
        ctx=ctx,
        lang=req.language or "en",
        file_bytes=file_bytes,
        file_mime=file_mime,
        file_name=file_name,
        file_desc=file_desc
    )

    reply_content = ai_result["reply"]
    is_emergency = ai_result.get("is_emergency", False)
    is_boundary = ai_result.get("prescription_boundary", False)
    citations = ai_result.get("citations", [])
    suggested_actions = ai_result.get("suggested_actions", [])
    model_used = ai_result.get("model_used", getattr(settings, "GEMINI_MODEL", "Gemini 3.8 Flash"))

    # Suggestion 6: Clinical Escalation when Emergency Red-Flag is Detected
    if is_emergency:
        try:
            from app.services.notification_service import notify_clinical_staff_emergency_chat
            escalate_pat = None
            if current_user.role == Role.PATIENT.value:
                escalate_pat = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
            elif target_patient_id:
                escalate_pat = db.get(Patient, target_patient_id)

            if escalate_pat:
                notify_clinical_staff_emergency_chat(
                    db=db,
                    patient=escalate_pat,
                    snippet=user_query
                )
        except Exception as e:
            logger.warning("Failed to escalate emergency chat to clinical staff: %s", e)

    # 5. Resolve or Create ChatSession for History Management
    session = None
    if req.session_id:
        session = db.get(ChatSession, req.session_id)
        if session and session.user_id != current_user.id:
            session = None

    if not session:
        # Generate initial title from query snippet
        clean_title = (user_query[:35] + "…") if len(user_query) > 35 else (user_query or "Medical Document Review")
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            patient_id=target_patient_id,
            role=current_user.role,
            title=clean_title,
            language=req.language or "en",
            created_at=dt.datetime.utcnow(),
            updated_at=dt.datetime.utcnow(),
        )
        db.add(session)
        db.flush()
    else:
        session.updated_at = dt.datetime.utcnow()
        if session.title in ("New Consultation", "New Chat") and user_query:
            session.title = (user_query[:35] + "…") if len(user_query) > 35 else user_query

    # 6. Save User & Assistant Messages
    user_msg_entry = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        sender="user",
        content=user_query or f"[Attached file: {file_name}]",
        attachment_info=attachment_record,
        created_at=dt.datetime.utcnow()
    )
    db.add(user_msg_entry)

    asst_msg_entry = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        sender="assistant",
        content=reply_content,
        citations=citations,
        is_emergency=is_emergency,
        prescription_boundary=is_boundary,
        model_used=model_used,
        suggested_actions=suggested_actions,
        created_at=dt.datetime.utcnow() + dt.timedelta(milliseconds=200)
    )
    db.add(asst_msg_entry)
    db.commit()

    return RecoveryChatResponse(
        session_id=session.id,
        reply=reply_content,
        answer=reply_content,
        is_emergency=is_emergency,
        emergency_escalation=is_emergency,
        prescription_boundary_enforced=is_boundary,
        grounded_source=f"Recovera Clinical Grounding ({model_used})",
        plan_found=ctx.get("has_plan", True),
        suggested_actions=suggested_actions,
        citations=citations,
        attending_doctor=ctx.get("attending_doctor"),
        model_used=model_used,
        metadata={"patient_id": target_patient_id, "role": current_user.role}
    )


@router.get("/api/recovery-assistant/context")
def get_recovery_context(
    patient_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns the aggregated live grounding context for inspection."""
    ctx = gather_clinical_context(db, current_user, patient_id)
    return {
        "status": "success",
        "context": ctx,
        "gemini_active": bool(get_gemini_api_key()),
    }


@router.post("/api/recovery-assistant/translate", response_model=ClinicalTranslateResponse)
def translate_clinical_messages(
    req: ClinicalTranslateRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Translates clinical messages on-demand across English, Tamil, and Hindi.
    Supports single message, list of texts, or keyed items with markdown formatting preservation.
    """
    raw_items: List[Dict[str, str]] = []
    if req.items:
        raw_items = [{"id": it.id or str(idx), "text": it.text} for idx, it in enumerate(req.items)]
    elif req.texts:
        raw_items = [{"id": str(idx), "text": t} for idx, t in enumerate(req.texts)]
    elif req.text:
        raw_items = [{"id": "0", "text": req.text}]

    if not raw_items:
        return ClinicalTranslateResponse(
            status="success",
            items=[],
            translations=[],
            translated_text=None,
            target_lang=req.target_lang
        )

    translated_items = translate_clinical_texts(
        items=raw_items,
        target_lang=req.target_lang,
        source_lang=req.source_lang
    )

    translations_list = [it["translated_text"] for it in translated_items]
    single_translated = translations_list[0] if translations_list else None

    return ClinicalTranslateResponse(
        status="success",
        items=translated_items,
        translations=translations_list,
        translated_text=single_translated,
        target_lang=req.target_lang
    )
