import uuid
import logging
import datetime as dt
from typing import Optional, List, Dict, Any
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.config import settings
from app.core.constants import SMSCategory, SMSJobStatus, Role
from app.models.notification import Notification
from app.models.sms_job import SMSJob
from app.models.user import User
from app.models.patient import Patient

logger = logging.getLogger("recovera.notifications")


def create_in_app_notification(
    db: Session,
    user_id: str,
    title: str,
    message: str,
    category: str = "general",
    link: Optional[str] = None
) -> Notification:
    """Create a persistent in-app notification for a user (localized if recipient is a patient)."""
    # Check if recipient is a patient with a preferred language
    patient = db.scalar(select(Patient).where(Patient.user_id == user_id))
    if patient:
        pref_lang = (patient.preferred_language or "en").lower().strip()[:2]
        if pref_lang in ("ta", "hi"):
            title = localize_in_app_text(title, pref_lang)
            message = localize_in_app_text(message, pref_lang)

    notif = Notification(
        user_id=user_id,
        title=title,
        message=message,
        category=category,
        link=link,
        is_read=False,
        created_at=dt.datetime.utcnow()
    )
    db.add(notif)
    db.flush()
    return notif


def localize_in_app_text(text: str, lang: str) -> str:
    """Localizes in-app notification title or message for patients into their preferred language."""
    if not text or not lang or lang not in ("ta", "hi"):
        return text

    if lang == "ta":
        replacements = [
            ("Appointment Cancelled by Doctor", "மருத்துவரால் சந்திப்பு ரத்து செய்யப்பட்டது"),
            ("Appointment Request Not Forwarded", "சந்திப்பு கோரிக்கை அனுப்பப்படவில்லை"),
            ("Appointment Cancelled", "சந்திப்பு ரத்து செய்யப்பட்டது"),
            ("Appointment Request Received", "சந்திப்பு கோரிக்கை பெறப்பட்டது"),
            ("Appointment Triage Completed", "சந்திப்பு பரிசீலனை முடிந்தது"),
            ("Appointment Reschedule Requested", "சந்திப்பு மறுதேதி கோரப்பட்டது"),
            ("Recovery Plan Published", "மீட்பு திட்டம் வெளியிடப்பட்டது"),
            ("Clinical Document Approved", "மருத்துவ ஆவணம் அங்கீகரிக்கப்பட்டது"),
            ("New Prescription Prescribed", "புதிய மருந்து பரிந்துரைக்கப்பட்டது"),
            ("Prescription Updated", "மருந்து பரிந்துரை புதுப்பிக்கப்பட்டது"),
            ("Medication Discontinued", "மருந்து நிறுத்தப்பட்டது"),
            ("Doctor Consultation Call", "மருத்துவர் ஆலோசனை அழைப்பு"),
            ("Daily Recovery Check-In", "தினசரி மீட்பு சரிபார்ப்பு"),
            ("Appointment Reminder", "மருத்துவ சந்திப்பு நினைவூட்டல்"),
            ("Medication Reminder", "மருந்து நினைவூட்டல்"),
            ("Medication Overdue", "மருந்து உட்கொள்ளும் நேரம் கடந்துவிட்டது"),
            ("was cancelled by the doctor.", "மருத்துவரால் ரத்து செய்யப்பட்டுள்ளது."),
            ("was cancelled by doctor.", "மருத்துவரால் ரத்து செய்யப்பட்டுள்ளது."),
            ("has been cancelled by the doctor.", "மருத்துவரால் ரத்து செய்யப்பட்டுள்ளது."),
            ("was not forwarded to", "செவிலியர் குழுவால் மருத்துவருக்கு அனுப்பப்படவில்லை:"),
            ("could not be forwarded to", "செவிலியர் குழுவால் மருத்துவருக்கு அனுப்பப்படவில்லை:"),
            ("by the clinical nursing staff.", "செவிலியர் குழுவால்."),
            ("by clinical staff.", "மருத்துவ ஊழியர்களால்."),
            ("Kindly retry or choose another available time slot", "தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது மாற்று நேரத்தைத் தேர்ந்தெடுக்கவும்"),
            ("Kindly retry or choose an alternative available slot", "தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது மாற்று நேரத்தைத் தேர்ந்தெடுக்கவும்"),
            ("You may request a new appointment anytime.", "நீங்கள் எப்போது வேண்டுமானாலும் புதிய சந்திப்பை கோரலாம்."),
            ("You may rebook in the portal.", "போர்ட்டலில் மீண்டும் பதிவு செய்யவும்."),
            ("Reason:", "காரணம்:"),
            ("has been scheduled for", "திட்டமிடப்பட்டுள்ளது:"),
            ("scheduled intake:", "திட்டமிடப்பட்ட உட்கொள்ளல்:"),
            ("Please take 1 minute to complete your daily recovery check-in for today.", "இன்றைய தினசரி மீட்பு சரிபார்ப்பை 1 நிமிடத்தில் பூர்த்தி செய்யவும்."),
        ]
        res = text
        for en, ta in replacements:
            res = res.replace(en, ta)
        return res

    elif lang == "hi":
        replacements = [
            ("Appointment Cancelled by Doctor", "डॉक्टर द्वारा अपॉइंटमेंट रद्द किया गया"),
            ("Appointment Request Not Forwarded", "अपॉइंटमेंट अनुरोध आगे नहीं भेजा गया"),
            ("Appointment Cancelled", "अपॉइंटमेंट रद्द किया गया"),
            ("Appointment Request Received", "अपॉइंटमेंट अनुरोध प्राप्त हुआ"),
            ("Appointment Triage Completed", "अपॉइंटमेंट समीक्षा पूरी हुई"),
            ("Appointment Reschedule Requested", "अपॉइंटमेंट पुनर्निर्धारण का अनुरोध किया गया"),
            ("Recovery Plan Published", "रिकवरी योजना प्रकाशित की गई"),
            ("Clinical Document Approved", "चिकित्सा दस्तावेज़ अनुमोदित"),
            ("New Prescription Prescribed", "नई दवा निर्धारित की गई"),
            ("Prescription Updated", "दवा अद्यतन की गई"),
            ("Medication Discontinued", "दवा बंद कर दी गई"),
            ("Doctor Consultation Call", "डॉक्टर परामर्श कॉल"),
            ("Daily Recovery Check-In", "दैनिक रिकवरी चेक-इन"),
            ("Appointment Reminder", "अपॉइंटमेंट अनुस्मारक"),
            ("Medication Reminder", "दवा अनुस्मारक"),
            ("Medication Overdue", "दवा का समय बीत चुका है"),
            ("was cancelled by the doctor.", "डॉक्टर द्वारा रद्द कर दिया गया है।"),
            ("was cancelled by doctor.", "डॉक्टर द्वारा रद्द कर दिया गया है।"),
            ("has been cancelled by the doctor.", "डॉक्टर द्वारा रद्द कर दिया गया है।"),
            ("was not forwarded to", "नर्सिंग स्टाफ द्वारा डॉक्टर को आगे नहीं भेजा गया:"),
            ("could not be forwarded to", "नर्सिंग स्टाफ द्वारा डॉक्टर को आगे नहीं भेजा गया:"),
            ("by the clinical nursing staff.", "नर्सिंग स्टाफ द्वारा।"),
            ("by clinical staff.", "अस्पताल कर्मचारियों द्वारा।"),
            ("Kindly retry or choose another available time slot", "कृपया पुन: प्रयास करें या अन्य समय स्लॉट चुनें"),
            ("Kindly retry or choose an alternative available slot", "कृपया पुन: प्रयास करें या वैकल्पिक स्लॉट चुनें"),
            ("You may request a new appointment anytime.", "आप किसी भी समय नया अपॉइंटमेंट बुक कर सकते हैं।"),
            ("You may rebook in the portal.", "कृपया पोर्टल में पुनः बुक करें।"),
            ("Reason:", "कारण:"),
            ("has been scheduled for", "निर्धारित किया गया है:"),
            ("scheduled intake:", "निर्धारित खुराक:"),
            ("Please take 1 minute to complete your daily recovery check-in for today.", "कृपया आज का दैनिक रिकवरी चेक-इन 1 मिनट में पूरा करें।"),
        ]
        res = text
        for en, hi in replacements:
            res = res.replace(en, hi)
        return res

    return text


def localize_sms_message(text: str, lang: str) -> str:
    """Localizes outgoing SMS alert text into the patient's preferred language (ta, hi, en)."""
    if not text or not lang or lang not in ("ta", "hi"):
        return text

    if lang == "ta":
        if "Medication reminder" in text or "medication reminder" in text.lower():
            t = text.replace("Medication reminder: please follow your approved prescription —", "மருந்து நினைவூட்டல்: உங்கள் மருந்தை உட்கொள்ளவும் —")
            t = t.replace("Medication reminder: Please take your scheduled", "மருந்து நினைவூட்டல்: திட்டமிடப்பட்ட மருந்தை உட்கொள்ளவும்:")
            t = t.replace("Open Recovera for your schedule.", "அட்டவணையைக் காண Recovera செயலியைத் திறக்கவும்.")
            t = t.replace("with food", "உணவுடன்").replace("before meals", "உணவுக்கு முன்").replace("after meals", "உணவுக்குப் பின்").replace("on empty stomach", "வெறும் வயிற்றில்")
            return t
        elif "We hope you have taken your" in text or "We hope you have taken your medicines" in text:
            return "உங்கள் மருந்துகளை உட்கொண்டிருப்பீர்கள் என நம்புகிறோம். Recovera செயலியைத் திறந்து மருந்து உட்கொண்ட நிலையைப் பதிவு செய்யவும்: http://localhost:5173"
        elif "scheduled intake for" in text.lower() or "overdue" in text.lower():
            t = text.replace("RecoveraAI Alert:", "RecoveraAI எச்சரிக்கை:")
            t = t.replace("is overdue. Please take your medicine or contact hospital nursing staff.", "நேரம் கடந்துவிட்டது. தயவுசெய்து மருந்தை உட்கொள்ளவும் அல்லது மருத்துவமனை ஊழியர்களைத் தொடர்பு கொள்ளவும்.")
            t = t.replace("RecoveraAI: You have an overdue dose of", "RecoveraAI: மருந்து உட்கொள்ளும் நேரம் கடந்துவிட்டது:")
            t = t.replace("If you are experiencing side effects or need help, contact your nursing team or open the app: http://localhost:5173", "உதவி தேவைப்பட்டால் செவிலியர் குழுவை தொடர்பு கொள்ளவும்: http://localhost:5173")
            return t
        elif "Daily Recovery Check-in" in text or "daily recovery check-in" in text.lower():
            return "RecoveraAI: இன்றைய தினசரி மீட்பு சரிபார்ப்பை (வலி அளவு, காய்ச்சல், இயக்கம்) பூர்த்தி செய்யவும்: http://localhost:5173/#/patient/checkin"
        # Context-aware: appointment cancelled by doctor
        elif "cancelled by the doctor" in text.lower() or "cancelled appointment on" in text.lower() or ("cancelled" in text.lower() and "doctor" in text.lower()):
            t = text.replace("RecoveraAI: Your appointment with", "RecoveraAI: மருத்துவ சந்திப்பு ரத்து —")
            t = t.replace("was cancelled by the doctor.", "மருத்துவரால் ரத்து செய்யப்பட்டுள்ளது.")
            t = t.replace("has been cancelled by the doctor.", "மருத்துவரால் ரத்து செய்யப்பட்டுள்ளது.")
            t = t.replace("cancelled appointment on", "சந்திப்பை ரத்து செய்தார்:")
            t = t.replace("Reason:", "காரணம்:").replace("Kindly retry in portal:", "தயவுசெய்து மீண்டும் முயற்சிக்கவும்:").replace("Please rebook in portal.", "தயவுசெய்து போர்ட்டலில் மீண்டும் பதிவு செய்யவும்.")
            return t
        # Context-aware: appointment not forwarded by nurse
        elif "not forwarded" in text.lower() or "could not be forwarded" in text.lower():
            t = text.replace("RecoveraAI: Your appointment request for", "RecoveraAI: சந்திப்பு கோரிக்கை —")
            t = t.replace("was not forwarded to", "அனுப்பப்படவில்லை:")
            t = t.replace("could not be forwarded to", "அனுப்பப்படவில்லை:")
            t = t.replace("by nursing staff.", "செவிலியர் குழுவால்.")
            t = t.replace("by the nursing staff.", "செவிலியர் குழுவால்.")
            t = t.replace("Reason:", "காரணம்:").replace("Kindly retry in portal:", "தயவுசெய்து மீண்டும் முயற்சிக்கவும்:")
            return t
        elif "forwarded to clinical staff" in text or "has been forwarded to clinical staff" in text:
            t = text.replace("RecoveraAI: Your appointment request with", "RecoveraAI: உங்கள் சந்திப்பு கோரிக்கை:")
            t = t.replace("has been forwarded to clinical staff. You will be notified once reviewed.", "செவிலியர் குழுவிற்கு அனுப்பப்பட்டுள்ளது. மதிப்பாய்வு செய்யப்பட்டதும் அறிவிக்கப்படுவீர்கள்.")
            return t
        elif "was triaged and forwarded to" in text:
            t = text.replace("RecoveraAI: Your appointment was triaged and forwarded to", "RecoveraAI: உங்கள் சந்திப்பு பரிசீலிக்கப்பட்டு அனுப்பப்பட்டது:")
            t = t.replace("for", "தேதி/நேரம்:")
            return t
        elif "upcoming consultation" in text or "upcoming appointment" in text or "Reminder: Your appointment with" in text or "Reminder — You have an upcoming appointment with" in text:
            t = text.replace("RecoveraAI: Reminder — You have an upcoming consultation with", "RecoveraAI: நினைவூட்டல் — மருத்துவ சந்திப்பு உள்ளது:")
            t = t.replace("RecoveraAI: Reminder — You have an upcoming appointment with", "RecoveraAI: நினைவூட்டல் — மருத்துவ சந்திப்பு உள்ளது:")
            t = t.replace("RecoveraAI Reminder: Your appointment with", "RecoveraAI நினைவூட்டல்: மருத்துவ சந்திப்பு உள்ளது —")
            t = t.replace("tomorrow at", "நாளை நேரம்:")
            t = t.replace("in 2 hours at", "இன்னும் 2 மணி நேரத்தில்:")
            t = t.replace("is scheduled for", "தேதி/நேரம்:")
            t = t.replace("Join at:", "இணைய:").replace("Details:", "விவரங்கள்:").replace("Open portal to view:", "போர்ட்டலில் பார்க்க:").replace("Open portal:", "போர்ட்டலில் பார்க்க:")
            return t
        elif "has been confirmed for" in text:
            t = text.replace("RecoveraAI: Your appointment with", "RecoveraAI: மருத்துவ சந்திப்பு உறுதி செய்யப்பட்டது:")
            t = t.replace("has been confirmed for", "தேதி/நேரம்:")
            t = t.replace("Details:", "விவரங்கள்:")
            return t
        elif "requested to reschedule your appointment to" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: டாக்டர்")
            t = t.replace("requested to reschedule your appointment to", "சந்திப்பை மாற்றக் கோரியுள்ளார்:")
            t = t.replace("Reason:", "காரணம்:")
            return t
        elif "Welcome to Recovera AI" in text:
            return "Recovera AI-க்கு நல்வரவு. உங்கள் மருத்துவமனை மீட்பு கண்காணிப்பு செயலில் உள்ளது. மருந்துகள் மற்றும் தினசரி சரிபார்ப்பை கண்காணிக்க: http://localhost:5173"
        elif "published your updated" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: டாக்டர்")
            t = t.replace("published your updated", "உங்கள் புதிய மீட்பு வழிகாட்டுதல்களை வெளியிட்டுள்ளார்:")
            t = t.replace("Please review your schedule and recovery guidelines at:", "போர்ட்டலில் பார்க்க:")
            return t
        elif "Please check your medication schedule:" in text:
            t = text.replace("RecoveraAI:", "RecoveraAI: மருந்து தகவல் —")
            t = t.replace("Please check your medication schedule:", "உங்கள் அட்டவணையை சரிபார்க்கவும்:")
            return t
        elif "is calling you for an immediate" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: டாக்டர்")
            t = t.replace("is calling you for an immediate", "உடனடி ஆலோசனைக்காக அழைக்கிறார்:")
            t = t.replace("consultation. Join now:", "இணைய:")
            return t
        elif "has been reviewed and approved by" in text:
            t = text.replace("RecoveraAI: Your latest", "RecoveraAI: உங்கள் சமீபத்திய")
            t = t.replace("has been reviewed and approved by", "சரிபார்க்கப்பட்டு அங்கீகரிக்கப்பட்டது:")
            t = t.replace("View in your portal:", "போர்ட்டலில் பார்க்க:")
            return t
        elif "Test message from Hospital SMS Gateway." in text:
            return "மருத்துவமனை எஸ்எம்எஸ் கேட்வேயில் இருந்து சோதனை செய்தி."

    elif lang == "hi":
        if "Medication reminder" in text or "medication reminder" in text.lower():
            t = text.replace("Medication reminder: please follow your approved prescription —", "दवा अनुस्मारक: कृपया अपनी निर्धारित दवा लें —")
            t = t.replace("Medication reminder: Please take your scheduled", "दवा अनुस्मारक: कृपया निर्धारित दवा लें:")
            t = t.replace("Open Recovera for your schedule.", "अपना शेड्यूल देखने के लिए Recovera खोलें।")
            t = t.replace("with food", "भोजन के साथ").replace("before meals", "भोजन से पहले").replace("after meals", "भोजन के बाद").replace("on empty stomach", "खाली पेट")
            return t
        elif "We hope you have taken your" in text or "We hope you have taken your medicines" in text:
            return "हमें उम्मीद है कि आपने दवा ले ली होगी। कृपया Recovera खोलें और दवा सेवन ट्रैकर भरें: http://localhost:5173"
        elif "scheduled intake for" in text.lower() or "overdue" in text.lower():
            t = text.replace("RecoveraAI Alert:", "RecoveraAI चेतावनी:")
            t = t.replace("is overdue. Please take your medicine or contact hospital nursing staff.", "समय बीत चुका है। कृपया दवा लें या अस्पताल स्टाफ से संपर्क करें।")
            t = t.replace("RecoveraAI: You have an overdue dose of", "RecoveraAI: आपकी दवा की खुराक का समय बीत चुका है:")
            t = t.replace("If you are experiencing side effects or need help, contact your nursing team or open the app: http://localhost:5173", "यदि सहायता चाहिए तो नर्सिंग स्टाफ से संपर्क करें: http://localhost:5173")
            return t
        elif "Daily Recovery Check-in" in text or "daily recovery check-in" in text.lower():
            return "RecoveraAI: कृपया आज के लिए अपना दैनिक रिकवरी चेक-इन (दर्द, बुखार, गतिशीलता) पूरा करें: http://localhost:5173/#/patient/checkin"
        # Context-aware: appointment cancelled by doctor
        elif "cancelled by the doctor" in text.lower() or "cancelled appointment on" in text.lower() or ("cancelled" in text.lower() and "doctor" in text.lower()):
            t = text.replace("RecoveraAI: Your appointment with", "RecoveraAI: अपॉइंटमेंट रद्द —")
            t = t.replace("was cancelled by the doctor.", "डॉक्टर द्वारा रद्द कर दिया गया है।")
            t = t.replace("has been cancelled by the doctor.", "डॉक्टर द्वारा रद्द कर दिया गया है।")
            t = t.replace("cancelled appointment on", "ने अपॉइंटमेंट रद्द किया:")
            t = t.replace("Reason:", "कारण:").replace("Kindly retry in portal:", "कृपया पुनः प्रयास करें:").replace("Please rebook in portal.", "कृपया पोर्टल में पुनः बुक करें।")
            return t
        # Context-aware: appointment not forwarded by nurse
        elif "not forwarded" in text.lower() or "could not be forwarded" in text.lower():
            t = text.replace("RecoveraAI: Your appointment request for", "RecoveraAI: अपॉइंटमेंट अनुरोध —")
            t = t.replace("was not forwarded to", "को आगे नहीं भेजा गया:")
            t = t.replace("could not be forwarded to", "को आगे नहीं भेजा गया:")
            t = t.replace("by nursing staff.", "नर्सिंग स्टाफ द्वारा।")
            t = t.replace("by the nursing staff.", "नर्सिंग स्टाफ द्वारा।")
            t = t.replace("Reason:", "कारण:").replace("Kindly retry in portal:", "कृपया पुनः प्रयास करें:")
            return t
        elif "forwarded to clinical staff" in text or "has been forwarded to clinical staff" in text:
            t = text.replace("RecoveraAI: Your appointment request with", "RecoveraAI: आपका अपॉइंटमेंट अनुरोध:")
            t = t.replace("has been forwarded to clinical staff. You will be notified once reviewed.", "नर्सिंग स्टाफ को भेज दिया गया है। समीक्षा के बाद सूचित किया जाएगा।")
            return t
        elif "was triaged and forwarded to" in text:
            t = text.replace("RecoveraAI: Your appointment was triaged and forwarded to", "RecoveraAI: आपका अपॉइंटमेंट प्रोसेस करके भेजा गया:")
            t = t.replace("for", "तारीख/समय:")
            return t
        elif "upcoming consultation" in text or "upcoming appointment" in text or "Reminder: Your appointment with" in text or "Reminder — You have an upcoming appointment with" in text:
            t = text.replace("RecoveraAI: Reminder — You have an upcoming consultation with", "RecoveraAI: अनुस्मारक — परामर्श अपॉइंटमेंट है:")
            t = t.replace("RecoveraAI: Reminder — You have an upcoming appointment with", "RecoveraAI: अनुस्मारक — अपॉइंटमेंट है:")
            t = t.replace("RecoveraAI Reminder: Your appointment with", "RecoveraAI अनुस्मारक: डॉक्टर के साथ अपॉइंटमेंट —")
            t = t.replace("tomorrow at", "कल समय:")
            t = t.replace("in 2 hours at", "2 घंटे में:")
            t = t.replace("is scheduled for", "तारीख/समय:")
            t = t.replace("Join at:", "शामिल हों:").replace("Details:", "विवरण:").replace("Open portal to view:", "पोर्टल में देखें:").replace("Open portal:", "पोर्टल में देखें:")
            return t
        elif "has been confirmed for" in text:
            t = text.replace("RecoveraAI: Your appointment with", "RecoveraAI: अपॉइंटमेंट कन्फर्म हो गई है:")
            t = t.replace("has been confirmed for", "तारीख/समय:")
            t = t.replace("Details:", "विवरण:")
            return t
        elif "requested to reschedule your appointment to" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: डॉ.")
            t = t.replace("requested to reschedule your appointment to", "ने अपॉइंटमेंट पुनर्निर्धारित करने का अनुरोध किया:")
            t = t.replace("Reason:", "कारण:")
            return t
        elif "Welcome to Recovera AI" in text:
            return "Recovera AI में आपका स्वागत है। आपकी अस्पताल रिकवरी निगरानी सक्रिय है। अपनी दवाएं और चेक-इन यहां ट्रैक करें: http://localhost:5173"
        elif "published your updated" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: डॉ.")
            t = t.replace("published your updated", "ने आपकी अद्यतन रिकवरी योजना प्रकाशित की है:")
            t = t.replace("Please review your schedule and recovery guidelines at:", "पोर्टल में देखें:")
            return t
        elif "Please check your medication schedule:" in text:
            t = text.replace("RecoveraAI:", "RecoveraAI: दवा सूचना —")
            t = t.replace("Please check your medication schedule:", "कृपया अपना शेड्यूल देखें:")
            return t
        elif "is calling you for an immediate" in text:
            t = text.replace("RecoveraAI: Dr.", "RecoveraAI: डॉ.")
            t = t.replace("is calling you for an immediate", "तत्काल परामर्श के लिए कॉल कर रहे हैं:")
            t = t.replace("consultation. Join now:", "शामिल हों:")
            return t
        elif "has been reviewed and approved by" in text:
            t = text.replace("RecoveraAI: Your latest", "RecoveraAI: आपका नवीनतम")
            t = t.replace("has been reviewed and approved by", "अनुमोदित कर दिया गया है:")
            t = t.replace("View in your portal:", "पोर्टल में देखें:")
            return t
        elif "Test message from Hospital SMS Gateway." in text:
            return "अस्पताल एसएमएस गेटवे से परीक्षण संदेश।"

    return text


def send_sms_job(
    db: Session,
    phone: str,
    message_text: str,
    category: str = SMSCategory.APPOINTMENT.value,
    patient_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    scheduled_for: Optional[dt.datetime] = None
) -> Optional[SMSJob]:
    """Queue and attempt delivery of an SMS via the Hospital SMS Gateway."""
    now = dt.datetime.utcnow()
    sched = scheduled_for or now
    key = idempotency_key or f"sms_{uuid.uuid4().hex}"
    existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == key))
    if existing:
        if existing.state == SMSJobStatus.QUEUED.value and sched <= now:
            from app.services.sms_gateway_service import dispatch_sms_job
            dispatch_sms_job(db, existing)
        return existing

    clean_phone = phone.strip()

    # Check patient status and language localization
    if patient_id:
        patient = db.get(Patient, patient_id)
        if patient:
            # If care status is idle, suppress routine non-emergency SMS reminders
            if getattr(patient, "care_status", "active") == "idle" and category in (
                SMSCategory.MEDICATION_REMINDER.value,
                SMSCategory.TRACKER_FOLLOWUP.value,
                SMSCategory.CHECKIN_REMINDER.value
            ):
                logger.info("Suppressing SMS reminder for idle patient %s", patient.patient_code)
                return None

            pref_lang = (getattr(patient, "sms_language", None) or patient.preferred_language or "en").lower().strip()[:2]
            message_text = localize_sms_message(message_text, pref_lang)
    elif clean_phone:
        # Fallback: look up patient by phone to ensure language localization is applied
        patient = db.scalar(select(Patient).where(Patient.phone == clean_phone))
        if patient:
            patient_id = patient.id
            pref_lang = (getattr(patient, "sms_language", None) or patient.preferred_language or "en").lower().strip()[:2]
            message_text = localize_sms_message(message_text, pref_lang)

    now = dt.datetime.utcnow()
    sched = scheduled_for or now

    job = SMSJob(
        patient_id=patient_id,
        recipient_phone=clean_phone,
        category=category,
        message_text=message_text,
        idempotency_key=key,
        state=SMSJobStatus.QUEUED.value,
        scheduled_for=sched,
        created_at=now
    )
    db.add(job)
    db.flush()

    if sched <= now:
        # Attempt immediate gateway delivery if configured
        if settings.ONEPLUS_GATEWAY_ENABLED and settings.ONEPLUS_GATEWAY_URL:
            gw_timeout = 0.3 if "test" in settings.DATABASE_URL.lower() else 3.0
            try:
                resp = requests.post(
                    f"{settings.ONEPLUS_GATEWAY_URL.rstrip('/')}/send",
                    json={
                        "phone": clean_phone,
                        "message": message_text,
                        "token": settings.ONEPLUS_GATEWAY_TOKEN
                    },
                    timeout=gw_timeout
                )
                job.last_attempt = dt.datetime.utcnow()
                if resp.status_code in (200, 201, 202):
                    job.state = SMSJobStatus.SENT.value
                    job.provider_message_id = resp.json().get("id", "gw_ack")
                else:
                    job.state = SMSJobStatus.FAILED.value
                    job.error_message = f"Gateway returned HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                logger.warning("SMS Gateway delivery attempt failed: %s", e)
                job.state = SMSJobStatus.FAILED.value
                job.last_attempt = dt.datetime.utcnow()
                job.error_message = str(e)
        else:
            # Gateway simulation mode: logged and marked sent for testing
            job.state = SMSJobStatus.SENT.value
            job.last_attempt = dt.datetime.utcnow()
            job.provider_message_id = f"sim_{uuid.uuid4().hex[:8]}"
    db.commit()

    return job


def notify_doctor_patient_assigned(db: Session, doctor_id: str, patient: Patient) -> None:
    """Alert doctor when a new patient is assigned to their care."""
    doctor = db.get(User, doctor_id)
    if not doctor:
        return

    patient_user = db.get(User, patient.user_id) if patient.user_id else None
    patient_name = patient_user.name if patient_user else patient.patient_code

    title = "New Patient Assigned"
    msg = f"Patient {patient_name} ({patient.patient_code}) has been assigned to your department."
    create_in_app_notification(
        db=db,
        user_id=doctor.id,
        title=title,
        message=msg,
        category="patient_assignment",
        link=f"#/staff/patients/{patient.id}"
    )

    if doctor.phone:
        send_sms_job(
            db=db,
            phone=doctor.phone,
            message_text=f"RecoveraAI: Patient {patient_name} ({patient.patient_code}) has been assigned to your care.",
            category=SMSCategory.REGISTRATION.value,
            patient_id=patient.id
        )


def notify_doctor_consultation_requested(
    db: Session,
    doctor_id: str,
    patient: Patient,
    mode: str,
    room_id: str,
    initiator_name: Optional[str] = None
) -> None:
    """Alert doctor when a patient or nurse requests/initiates a teleconsultation."""
    doctor = db.get(User, doctor_id)
    if not doctor:
        return

    patient_user = db.get(User, patient.user_id) if patient.user_id else None
    patient_name = patient_user.name if patient_user else patient.patient_code

    base_url = getattr(settings, "FRONTEND_URL", "http://localhost:5173").rstrip('/')
    if initiator_name:
        title = f"Telehealth Bridge: {initiator_name}"
        msg = f"{initiator_name} initiated an immediate {mode.upper()} consultation with Patient {patient_name}. Click to join room."
        sms_msg = f"RecoveraAI: {initiator_name} connected you with Patient {patient_name} for an immediate {mode} consultation. Join room: {base_url}/#/consultation/{room_id}"
    else:
        title = "New Consultation Request"
        msg = f"Patient {patient_name} requested an immediate {mode.upper()} consultation. Click to join room."
        sms_msg = f"RecoveraAI: Patient {patient_name} requested an immediate {mode} consultation. Join room: {base_url}/#/consultation/{room_id}"

    create_in_app_notification(
        db=db,
        user_id=doctor.id,
        title=title,
        message=msg,
        category="consultation",
        link=f"#/consultation/{room_id}"
    )

    if doctor.phone:
        send_sms_job(
            db=db,
            phone=doctor.phone,
            message_text=sms_msg,
            category=SMSCategory.CONSULTATION.value,
            patient_id=patient.id
        )


def notify_patient_consultation_called(db: Session, patient: Patient, doctor: User, room_id: str, mode: str) -> None:
    """Notify patient when their attending doctor calls / books an immediate consultation."""
    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Doctor Consultation Call",
            message=f"Dr. {doctor.name} is calling you for an immediate {mode.upper()} consultation. Click to join.",
            category="consultation",
            link=f"#/consultation/{room_id}"
        )

    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: Dr. {doctor.name} is calling you for an immediate {mode} consultation. Join now: http://localhost:5173/#/consultation/{room_id}",
            category=SMSCategory.CONSULTATION.value,
            patient_id=patient.id
        )


# Backward-compatible alias
notify_patient_consultation_opened = notify_patient_consultation_called


def notify_doctor_appointment_booked(db: Session, doctor_id: str, patient: Patient, start_time: dt.datetime) -> None:
    """Alert doctor when an appointment is booked."""
    doctor = db.get(User, doctor_id)
    if not doctor:
        return

    patient_user = db.get(User, patient.user_id) if patient.user_id else None
    patient_name = patient_user.name if patient_user else patient.patient_code
    time_str = start_time.strftime("%d %b %Y at %I:%M %p")

    title = "New Appointment Scheduled"
    msg = f"New appointment scheduled with {patient_name} for {time_str}."
    create_in_app_notification(
        db=db,
        user_id=doctor.id,
        title=title,
        message=msg,
        category="appointment",
        link="#/staff/appointments"
    )

    if doctor.phone:
        send_sms_job(
            db=db,
            phone=doctor.phone,
            message_text=f"RecoveraAI: Appointment scheduled with {patient_name} for {time_str}.",
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id
        )


def notify_nurses_appointment_requested(db: Session, appointment, patient: Patient) -> None:
    """Alert all active nurses and acknowledge patient when an appointment request is submitted needing triage & forwarding."""
    patient_user = db.get(User, patient.user_id) if patient.user_id else None
    patient_name = patient_user.name if patient_user else patient.patient_code
    doctor = db.get(User, appointment.doctor_id) if getattr(appointment, "doctor_id", None) else None
    doctor_name = doctor.name if doctor else "Doctor"
    start_dt = getattr(appointment, "start_time", None)
    time_str = start_dt.strftime("%d %b %Y at %I:%M %p") if start_dt else "requested time"

    # 1. Acknowledge Patient: Request received and forwarded to clinical staff (prevent premature confirmation)
    if patient_user:
        create_in_app_notification(
            db=db,
            user_id=patient_user.id,
            title="Appointment Request Received",
            message=f"Your appointment request with Dr. {doctor_name} for {time_str} has been received and forwarded to clinical staff for nurse triage.",
            category="appointment",
            link="#/patient/appointments"
        )
        if patient.phone and patient.sms_opt_in:
            send_sms_job(
                db=db,
                phone=patient.phone,
                message_text=f"RecoveraAI: Your appointment request with Dr. {doctor_name} ({time_str}) has been forwarded to clinical staff. You will be notified once reviewed.",
                category=SMSCategory.APPOINTMENT.value,
                patient_id=patient.id
            )

    # 2. Alert all active nurses
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    for n in nurses:
        create_in_app_notification(
            db=db,
            user_id=n.id,
            title="New Patient Appointment Triage",
            message=f"Patient {patient_name} submitted an appointment request with Dr. {doctor_name} for {time_str} requiring clinical triage and doctor assignment.",
            category="appointment",
            link="#/staff/requests"
        )
        if n.phone:
            send_sms_job(
                db=db,
                phone=n.phone,
                message_text=f"RecoveraAI: New appointment request from patient {patient_name} with Dr. {doctor_name} ({time_str}) requiring nurse triage.",
                category=SMSCategory.APPOINTMENT.value,
                patient_id=patient.id
            )


def cancel_pending_appointment_notifications(db: Session, appointment_id: str) -> int:
    """Cancels queued SMS reminder jobs associated with an appointment when it is cancelled or discontinued."""
    jobs = db.scalars(
        select(SMSJob).where(
            SMSJob.idempotency_key.like(f"%appt_%_{appointment_id}%"),
            SMSJob.state == SMSJobStatus.QUEUED.value
        )
    ).all()
    count = 0
    for job in jobs:
        job.state = SMSJobStatus.CANCELLED.value
        count += 1
    return count


def notify_nurse_reschedule_requested(
    db: Session,
    appointment,
    doctor: User,
    reschedule_date: dt.datetime,
    reason: Optional[str] = None
) -> None:
    """
    Alert the assigning nurse (and nursing pool) when a doctor requests an appointment reschedule.
    Requirement:
      'saying doctor [x] is asking for a reschedule on {date}reason:{not specified} or { the custom message if mentioned by doctor}'
    """
    date_str = reschedule_date.strftime("%d %b %Y at %I:%M %p")
    reason_clean = reason.strip() if reason and reason.strip() else "not specified"
    sms_body = f"doctor [{doctor.name}] is asking for a reschedule on {date_str} reason:{reason_clean}"

    # 1. Notify the booking staff member (nurse/admin)
    booking_user = db.get(User, appointment.booked_by)
    target_users = set()
    if booking_user:
        target_users.add(booking_user.id)

    # Also notify active nurses so the triage floor is informed
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    for n in nurses:
        target_users.add(n.id)

    title = "Doctor Reschedule Request"
    in_app_msg = f"Doctor [{doctor.name}] is asking for a reschedule on {date_str} reason: {reason_clean}"

    for uid in target_users:
        create_in_app_notification(
            db=db,
            user_id=uid,
            title=title,
            message=in_app_msg,
            category="reschedule_request",
            link="#/staff/appointments"
        )

    # 2. Dispatch SMS Gateway alert to the booking nurse if phone exists
    if booking_user and booking_user.phone:
        send_sms_job(
            db=db,
            phone=booking_user.phone,
            message_text=f"RecoveraAI: {sms_body}",
            category=SMSCategory.APPOINTMENT.value,
            patient_id=appointment.patient_id
        )

    # Also inform the patient
    patient = db.get(Patient, appointment.patient_id)
    if patient and patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Appointment Reschedule Requested",
            message=f"Dr. [{doctor.name}] requested to reschedule your appointment to {date_str}. Nursing staff will confirm with you shortly.",
            category="appointment",
            link="#/patient/appointments"
        )
        if patient.phone and patient.sms_opt_in:
            send_sms_job(
                db=db,
                phone=patient.phone,
                message_text=f"RecoveraAI: Dr. {doctor.name} requested to reschedule your appointment to {date_str}. Reason: {reason_clean}",
                category=SMSCategory.APPOINTMENT.value,
                patient_id=patient.id
            )


# ===========================================================================
# 7 CLINICAL EXTENSIONS (IN-APP + SMS GATEWAY)
# ===========================================================================

def notify_clinical_staff_critical_checkin(db: Session, patient: Patient, checkin) -> None:
    """
    Suggestion 1: Critical Vitals / High-Risk Check-In Alert.
    Triggered when patient reports pain >= 7 or fever in daily check-in.
    Notifies attending doctor and floor nurses via In-App Alert and SMS Gateway.
    """
    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code
    fever_str = "with Fever" if checkin.fever else "without Fever"
    loc_str = f" ({checkin.pain_location})" if checkin.pain_location else ""

    title = "⚠️ High-Risk Check-In Alert"
    msg = f"Patient {pat_name} reported severe Pain: {checkin.pain_level}/10{loc_str} {fever_str}. Immediate clinical review required."

    # In-App alert to assigned doctor
    target_ids = set()
    if patient.assigned_doctor_id:
        target_ids.add(patient.assigned_doctor_id)

    # In-App alert to active nursing team
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    for n in nurses:
        target_ids.add(n.id)

    for uid in target_ids:
        create_in_app_notification(
            db=db,
            user_id=uid,
            title=title,
            message=msg,
            category="critical_vitals",
            link=f"#/staff/patients/{patient.id}/overview"
        )

    # SMS Gateway alert to attending doctor
    if patient.assigned_doctor_id:
        doc = db.get(User, patient.assigned_doctor_id)
        if doc and doc.phone:
            send_sms_job(
                db=db,
                phone=doc.phone,
                message_text=f"RecoveraAI CRITICAL: Patient {pat_name} submitted high-risk vitals (Pain {checkin.pain_level}/10, Fever: {'Yes' if checkin.fever else 'No'}). Review at: http://localhost:5173/#/staff/patients/{patient.id}/overview",
                category=SMSCategory.CHECKIN_REMINDER.value,
                patient_id=patient.id
            )
    db.commit()


def notify_patient_evening_checkin_reminder(db: Session, patient: Patient, idempotency_key: Optional[str] = None) -> None:
    """
    Suggestion 2: Evening Daily Check-In Reminder.
    Dispatched when an active patient has not submitted their check-in for the day.
    """
    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code

    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Daily Recovery Check-In",
            message="Please take 1 minute to complete your daily recovery check-in for today.",
            category="checkin_reminder",
            link="#/patient/checkin"
        )

    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: Hi {pat_name}, please take 1 minute to complete your daily recovery check-in for today: http://localhost:5173/#/patient/checkin",
            category=SMSCategory.CHECKIN_REMINDER.value,
            patient_id=patient.id,
            idempotency_key=idempotency_key
        )
    elif idempotency_key:
        marker = SMSJob(
            patient_id=patient.id,
            recipient_phone=patient.phone or "N/A",
            category=SMSCategory.CHECKIN_REMINDER.value,
            message_text="In-app check-in reminder recorded (SMS opt-in false)",
            idempotency_key=idempotency_key,
            state=SMSJobStatus.SENT.value,
            scheduled_for=dt.datetime.utcnow(),
            created_at=dt.datetime.utcnow()
        )
        db.add(marker)
        db.flush()
    db.commit()


def notify_staff_and_patient_missed_medication(
    db: Session,
    patient: Patient,
    med_name: str,
    dose_str: str,
    sched_time_str: str,
    idempotency_key: Optional[str] = None
) -> None:
    """
    Suggestion 3: Missed Medication Escalation (T + 2 Hours).
    Dispatched if medication remains unconfirmed 2 hours post-scheduled intake.
    """
    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code

    # In-App alert to nurses & attending doctor
    target_ids = set()
    if patient.assigned_doctor_id:
        target_ids.add(patient.assigned_doctor_id)
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    for n in nurses:
        target_ids.add(n.id)

    title = "Medication Overdue (2 Hours)"
    msg = f"Patient {pat_name} has not confirmed {med_name} ({dose_str}) scheduled at {sched_time_str}. Nursing outreach recommended."

    for uid in target_ids:
        create_in_app_notification(
            db=db,
            user_id=uid,
            title=title,
            message=msg,
            category="medication_overdue",
            link=f"#/staff/patients/{patient.id}/medications"
        )

    # SMS Gateway final reminder to patient
    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: You have an overdue dose of {med_name}. If you are experiencing side effects or need help, contact your nursing team or open the app: http://localhost:5173",
            category=SMSCategory.MEDICATION_REMINDER.value,
            patient_id=patient.id,
            idempotency_key=idempotency_key
        )
    db.commit()


def notify_patient_recovery_plan_updated(
    db: Session,
    patient: Patient,
    doctor: User,
    plan_title: str = "Recovery Plan"
) -> None:
    """
    Suggestion 4: Recovery Plan or Guideline Published / Updated.
    Dispatched when a physician publishes or updates recovery plan guidelines.
    """
    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Recovery Plan Published",
            message=f"Dr. {doctor.name} has published your updated {plan_title} and care instructions.",
            category="recovery_plan",
            link="#/patient/recovery-plan"
        )

    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: Dr. {doctor.name} published your updated {plan_title}. Please review your schedule and recovery guidelines at: http://localhost:5173",
            category=SMSCategory.REGISTRATION.value,
            patient_id=patient.id
        )
    db.commit()


def notify_patient_medication_prescribed(
    db: Session,
    patient: Patient,
    doctor: User,
    med_name: str,
    dose: str,
    action: str = "prescribed"
) -> None:
    """
    Suggestion 4 (Medications): Dispatched when a new medication is prescribed, updated, or discontinued.
    """
    if action == "prescribed":
        action_verb = "added to your active prescriptions"
        action_title = "New Prescription Prescribed"
    elif action == "updated":
        action_verb = "updated your prescription for"
        action_title = "Prescription Updated"
    else:
        action_verb = "discontinued"
        action_title = "Medication Discontinued"

    clinician_label = (
        f"Nurse {doctor.name}" if getattr(doctor, 'role', '') == Role.NURSE.value
        else (doctor.name if doctor.name.startswith("Dr.") else f"Dr. {doctor.name}")
    )

    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title=action_title,
            message=f"{clinician_label} has {action_verb}: {med_name} ({dose}).",
            category="prescription",
            link="#/patient/medications"
        )

    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: {clinician_label} has {action_verb}: {med_name} ({dose}). Please check your medication schedule: http://localhost:5173",
            category=SMSCategory.MEDICATION_REMINDER.value,
            patient_id=patient.id
        )
    db.commit()


def notify_appointment_upcoming_reminder(
    db: Session,
    appointment,
    hours_before: int,
    idempotency_key: Optional[str] = None
) -> None:
    """
    Suggestion 5: Upcoming Appointment Reminder (24h or 2h prior).
    Dispatches In-App alerts to Doctor and Patient, and SMS Gateway to Patient.
    """
    patient = db.get(Patient, appointment.patient_id)
    doctor = db.get(User, appointment.doctor_id)
    if not patient or not doctor:
        return

    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code
    time_str = appointment.start_time.strftime("%I:%M %p")
    date_str = appointment.start_time.strftime("%d %b %Y")
    key = idempotency_key or f"appt_remind_{hours_before}h_{appointment.id}"

    # Patient notifications
    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title=f"Appointment Reminder ({hours_before}h)",
            message=f"Reminder: You have an appointment with Dr. {doctor.name} on {date_str} at {time_str}.",
            category="appointment",
            link="#/patient/appointments"
        )

    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI Reminder: Your appointment with Dr. {doctor.name} is scheduled for {date_str} at {time_str}. Open portal to view: http://localhost:5173",
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id,
            idempotency_key=key
        )
    elif key:
        marker = SMSJob(
            patient_id=patient.id,
            recipient_phone=patient.phone or "N/A",
            category=SMSCategory.APPOINTMENT.value,
            message_text=f"In-app appointment reminder recorded ({hours_before}h)",
            idempotency_key=key,
            state=SMSJobStatus.SENT.value,
            scheduled_for=dt.datetime.utcnow(),
            created_at=dt.datetime.utcnow()
        )
        db.add(marker)
        db.flush()

    # Doctor In-App notification
    create_in_app_notification(
        db=db,
        user_id=doctor.id,
        title=f"Upcoming Appointment ({hours_before}h)",
        message=f"You have an upcoming appointment with {pat_name} on {date_str} at {time_str}.",
        category="appointment",
        link="#/staff/appointments"
    )
    db.commit()


def notify_clinical_staff_emergency_chat(
    db: Session,
    patient: Patient,
    snippet: str
) -> None:
    """
    Suggestion 6: AI Chatbot Emergency Red-Flag Clinical Escalation.
    Dispatched when patient reports acute chest pain, dyspnea, or hemorrhage in AI chat.
    Alerts attending physician and nursing team immediately.
    """
    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code
    clean_snippet = snippet[:120].strip()

    title = "🚨 Emergency Red-Flag in AI Chat"
    msg = f"Patient {pat_name} reported acute symptoms in Recovery AI Chat: \"{clean_snippet}\". Immediate clinical review required."

    target_ids = set()
    if patient.assigned_doctor_id:
        target_ids.add(patient.assigned_doctor_id)
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    for n in nurses:
        target_ids.add(n.id)

    for uid in target_ids:
        create_in_app_notification(
            db=db,
            user_id=uid,
            title=title,
            message=msg,
            category="emergency_escalation",
            link=f"#/staff/patients/{patient.id}/overview"
        )

    # SMS Gateway to assigned doctor
    if patient.assigned_doctor_id:
        doc = db.get(User, patient.assigned_doctor_id)
        if doc and doc.phone:
            send_sms_job(
                db=db,
                phone=doc.phone,
                message_text=f"RecoveraAI URGENT: Patient {pat_name} triggered emergency triage in AI chat (\"{clean_snippet[:80]}\"). Check record: http://localhost:5173",
                category=SMSCategory.CONSULTATION.value,
                patient_id=patient.id
            )
    db.commit()


def notify_document_approved(
    db: Session,
    doc,
    doctor: User
) -> None:
    """
    Suggestion 7: Clinical Document / Lab Report Reviewed & Approved.
    Dispatches In-App notification to uploading nurse and In-App + Gateway SMS to patient.
    """
    patient = db.get(Patient, doc.patient_id)
    if not patient:
        return

    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    pat_name = pat_user.name if pat_user else patient.patient_code
    doc_label = doc.doc_type.replace("_", " ").title() if doc.doc_type else "Clinical Document"

    # 1. In-App alert to uploading / reviewing nurse
    if doc.uploaded_by and doc.uploaded_by != doctor.id:
        uploader = db.get(User, doc.uploaded_by)
        if uploader:
            create_in_app_notification(
                db=db,
                user_id=uploader.id,
                title="Document Approved by Physician",
                message=f"Dr. {doctor.name} approved {doc_label} for patient {pat_name}.",
                category="document",
                link=f"#/staff/patients/{doc.patient_id}/documents"
            )

    # 2. In-App alert to patient
    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Clinical Document Approved",
            message=f"Dr. {doctor.name} reviewed and approved your {doc_label}.",
            category="document",
            link="#/patient/documents"
        )

    # 3. SMS Gateway alert to patient
    if patient.phone and patient.sms_opt_in:
        send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=f"RecoveraAI: Your latest {doc_label} has been reviewed and approved by Dr. {doctor.name}. View in your portal: http://localhost:5173",
            category=SMSCategory.REGISTRATION.value,
            patient_id=patient.id
        )
    db.commit()


def schedule_appointment_patient_notifications(
    db: Session,
    appointment,
) -> Dict[str, Any]:
    """
    Schedules 3 notification / SMS schedules for the patient when a follow-up or consultation is scheduled:
    1. Immediate Notification + SMS: Dispatched at booking time to confirm the schedule.
    2. 24-Hour (1 day) Prior Reminder: Queued to alert patient the day before the schedule.
    3. 2-Hour Prior Reminder: Queued to alert patient 2 hours prior to the schedule.
    """
    patient = db.get(Patient, appointment.patient_id)
    doctor = db.get(User, appointment.doctor_id)
    if not patient or not doctor:
        return {"immediate": None, "reminder_24h": None, "reminder_2h": None}

    start_dt = appointment.start_time
    date_str = start_dt.strftime("%d %b %Y")
    time_str = start_dt.strftime("%I:%M %p")
    now = dt.datetime.utcnow()

    is_follow_up = getattr(appointment, "appointment_type", "") == "follow_up"
    type_name = "Follow-up Appointment" if is_follow_up else "Appointment"

    # --- SCHEDULE 1: Immediate in-app confirmation & SMS (when scheduled) ---
    if patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title=f"{type_name} Scheduled",
            message=f"Your {type_name.lower()} with Dr. {doctor.name} has been scheduled for {date_str} at {time_str}.",
            category="appointment",
            link="#/patient/appointments"
        )

    job1 = None
    if patient.phone:
        sms_msg1 = f"RecoveraAI: Your appointment with Dr. {doctor.name} has been confirmed for {date_str} at {time_str}. Details: http://localhost:5173"
        job1 = send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=sms_msg1,
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id,
            idempotency_key=f"appt_confirm_{appointment.id}",
            scheduled_for=now
        )

    # --- SCHEDULE 2: Reminder the day before the schedule (24 hours prior) ---
    sched_24h = start_dt - dt.timedelta(days=1)
    job2 = None
    if patient.phone:
        sms_msg2 = f"RecoveraAI: Reminder — You have an upcoming appointment with Dr. {doctor.name} tomorrow at {time_str}. Open portal: http://localhost:5173"
        job2 = send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=sms_msg2,
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id,
            idempotency_key=f"appt_remind_24h_{appointment.id}",
            scheduled_for=sched_24h
        )

    # --- SCHEDULE 3: Reminder 2 hours prior to the schedule ---
    sched_2h = start_dt - dt.timedelta(hours=2)
    job3 = None
    if patient.phone:
        sms_msg3 = f"RecoveraAI: Reminder — You have an upcoming appointment with Dr. {doctor.name} in 2 hours at {time_str}. Open portal: http://localhost:5173"
        job3 = send_sms_job(
            db=db,
            phone=patient.phone,
            message_text=sms_msg3,
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id,
            idempotency_key=f"appt_remind_2h_{appointment.id}",
            scheduled_for=sched_2h
        )

    return {
        "immediate": job1,
        "reminder_24h": job2,
        "reminder_2h": job3
    }



