import React, { useState, useEffect } from 'react';
import {
  CalendarDays, CheckCircle2, ChevronRight, Activity, HeartPulse,
  Clock, Award, ShieldCheck, Sparkles
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function PatientRecoveryJourneyCard({ patientId, session, lang = 'en', onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = session?.token || JSON.parse(localStorage.getItem('recovera_session') || '{}')?.token;
    if (!patientId || !token) return;

    fetch(`${API_BASE}/api/digital-twin/patients/${patientId}/trajectory`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    })
      .then(res => res.ok ? res.json() : null)
      .then(json => {
        if (json) setData(json);
      })
      .catch(err => console.error('Error fetching patient trajectory:', err))
      .finally(() => setLoading(false));
  }, [patientId, session?.token]);

  if (loading || !data) return null;

  const currentDay = data.current_day != null ? Math.max(1, Math.min(14, Number(data.current_day))) : 1;
  const progressPct = Math.min(100, Math.round((currentDay / 14) * 100));

  const tier = (data.deviation_tier || 'normal').toLowerCase();
  let badgeKey = 'on_track';
  let badgeClass = 'badge-green';

  if (tier.includes('critical')) {
    badgeKey = 'critical';
    badgeClass = 'badge-red';
  } else if (tier.includes('moderate')) {
    badgeKey = 'moderate';
    badgeClass = 'badge-amber';
  } else if (tier.includes('mild')) {
    badgeKey = 'mild';
    badgeClass = 'badge-blue';
  }

  const badgeLabels = {
    en: {
      on_track: '● Healing Progress: On Track',
      mild: '● Healing Progress: Mild Deviation',
      moderate: '● Healing Trajectory: Moderate Deviation',
      critical: '● Healing Trajectory: Action Recommended'
    },
    ta: {
      on_track: '● குணமடைதல் முன்னேற்றம்: சரியான பாதையில்',
      mild: '● குணமடைதல் முன்னேற்றம்: லேசான விலகல்',
      moderate: '● குணமடைதல் முன்னேற்றம்: மிதமான விலகல்',
      critical: '● குணமடைதல் முன்னேற்றம்: தீவிர விலகல்'
    },
    hi: {
      on_track: '● रिकवरी प्रगति: सही राह पर',
      mild: '● रिकवरी प्रगति: हल्का विचलन',
      moderate: '● रिकवरी प्रगति: मध्यम विचलन',
      critical: '● रिकवरी प्रगति: तत्काल ध्यान आवश्यक'
    }
  };

  const contentMap = {
    en: {
      badge: (badgeLabels.en[badgeKey] || badgeLabels.en.on_track),
      pathway: '14-Day Post-Discharge Pathway',
      journeyTitle: `My Recovery Journey — Day ${currentDay} of 14`,
      checkinBtn: 'Complete Daily Check-in',
      corridorLabel: 'Healing Corridor Progress',
      completed: 'Completed',
      currentLabel: 'CURRENT',
      milestones: [
        { day: 'Day 1–3', title: 'Initial Healing & Rest', desc: 'Surgical recovery, medication start, and close monitoring' },
        { day: 'Day 4–7', title: 'Early Mobility', desc: 'Gradual reduction in discomfort, light movement exercises' },
        { day: 'Day 8–11', title: 'Progressive Healing', desc: 'Strengthening routine, steady pain tapering, daily check-ins' },
        { day: 'Day 12–14', title: 'Full Recovery Transition', desc: 'Returning to regular daily activities and final clinical review' }
      ]
    },
    ta: {
      badge: (badgeLabels.ta[badgeKey] || badgeLabels.ta.on_track),
      pathway: '14-நாள் மருத்துவமனைக்கு பிந்தைய மீட்புப் பாதை',
      journeyTitle: `எனது மீட்புப் பயணம் — நாள் ${currentDay} / 14`,
      checkinBtn: 'தினசரி சோதனையை முடிக்கவும்',
      corridorLabel: 'குணமடைதல் முன்னேற்ற நிலை',
      completed: 'முடிந்தது',
      currentLabel: 'தற்போதைய நிலை',
      milestones: [
        { day: 'நாள் 1–3', title: 'ஆரம்ப மீட்பு & ஓய்வு', desc: 'அறுவை சிகிச்சை மீட்பு, மருந்துகள் தொடக்கம் மற்றும் தீவிர கண்காணிப்பு' },
        { day: 'நாள் 4–7', title: 'ஆரம்ப நடமாட்டம்', desc: 'வலி குறைதல், மெதுவான நடைபயிற்சி மற்றும் லேசான செயல்பாடுகள்' },
        { day: 'நாள் 8–11', title: 'படிப்படியான முன்னேற்றம்', desc: 'வலுப்படுத்தும் பயிற்சிகள், வலி தணிவு மற்றும் தினசரி சோதனை' },
        { day: 'நாள் 12–14', title: 'முழு மீட்பு மாற்றம்', desc: 'வழக்கமான தினசரி பணிகளுக்கு திரும்புதல் மற்றும் இறுதி மருத்துவ ஆய்வு' }
      ]
    },
    hi: {
      badge: (badgeLabels.hi[badgeKey] || badgeLabels.hi.on_track),
      pathway: '14-दिवसीय डिस्चार्ज-उपरांत रिकवरी मार्ग',
      journeyTitle: `मेरी रिकवरी यात्रा — दिन ${currentDay} / 14`,
      checkinBtn: 'दैनिक चेक-इन पूरा करें',
      corridorLabel: 'रिकवरी प्रगति स्तर',
      completed: 'पूर्ण',
      currentLabel: 'वर्तमान चरण',
      milestones: [
        { day: 'दिन 1–3', title: 'प्रारंभिक आराम और उपचार', desc: 'सर्जरी के बाद आराम, दवाइयां शुरू करना और करीबी निगरानी' },
        { day: 'दिन 4–7', title: 'हल्की गतिशीलता', desc: 'दर्द में धीरे-धीरे कमी, हल्का चलना और रिकवरी व्यायाम' },
        { day: 'दिन 8–11', title: 'प्रगतिशील सुधार', desc: 'दैनिक चेक-इन, दर्द में कमी और नियमित सक्रिय प्रोटोकॉल' },
        { day: 'दिन 12–14', title: 'पूर्ण रिकवरी संक्रमण', desc: 'दैनिक गतिविधियों में वापसी और अंतिम क्लिनिकल समीक्षा' }
      ]
    }
  };

  const t = contentMap[lang] || contentMap.en;

  return (
    <div className="card" style={{ padding: 22, marginTop: 18, background: 'var(--surface)', border: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 44,
            height: 44,
            borderRadius: 12,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, rgba(0,212,184,0.18), rgba(56,189,248,0.18))',
            color: 'var(--p)'
          }}>
            <HeartPulse size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`badge ${badgeClass}`} style={{ fontSize: 11, fontWeight: 700 }}>
                {t.badge}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {t.pathway}
              </span>
            </div>
            <h3 style={{ margin: '3px 0 0', fontSize: 16, fontWeight: 800, color: 'var(--text)' }}>
              {t.journeyTitle}
            </h3>
          </div>
        </div>

        <button
          className="btn btn-primary btn-sm"
          onClick={() => onNavigate ? onNavigate('#/patient/check-in') : (window.location.hash = '#/patient/check-in')}
          style={{ fontSize: 12 }}
        >
          <CheckCircle2 size={13} /> {t.checkinBtn}
        </button>
      </div>

      {/* Progress Bar */}
      <div style={{ marginTop: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
          <span style={{ color: 'var(--text-dim)', fontWeight: 600 }}>{t.corridorLabel}</span>
          <span style={{ color: 'var(--p)', fontWeight: 700 }}>{progressPct}% {t.completed}</span>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: 'var(--surface2)', overflow: 'hidden' }}>
          <div style={{
            width: `${progressPct}%`,
            height: '100%',
            borderRadius: 4,
            background: 'linear-gradient(90deg, var(--p) 0%, #38bdf8 100%)',
            boxShadow: '0 0 10px rgba(0,212,184,0.4)',
            transition: 'width 0.4s ease'
          }} />
        </div>
      </div>

      {/* 4-Stage Milestone Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 150px), 1fr))', gap: 10, marginTop: 16 }}>
        {t.milestones.map((m, idx) => {
          const isDone = (idx + 1) * 3.5 <= currentDay;
          const isCurrent = !isDone && (idx * 3.5 <= currentDay);

          return (
            <div key={m.day} style={{
              padding: 12,
              borderRadius: 8,
              background: isCurrent ? 'rgba(0, 212, 184, 0.08)' : 'var(--surface2)',
              border: `1px solid ${isCurrent ? 'var(--p)' : 'var(--border)'}`,
              display: 'flex',
              flexDirection: 'column',
              gap: 4
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: isCurrent ? 'var(--p)' : 'var(--text-muted)' }}>
                  {m.day}
                </span>
                {isDone ? (
                  <CheckCircle2 size={13} style={{ color: 'var(--success)' }} />
                ) : isCurrent ? (
                  <span className="badge badge-teal" style={{ fontSize: 9 }}>{t.currentLabel}</span>
                ) : null}
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>
                {m.title}
              </div>
              <p style={{ margin: 0, fontSize: 11.5, color: 'var(--text-dim)', lineHeight: 1.4 }}>
                {m.desc}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
