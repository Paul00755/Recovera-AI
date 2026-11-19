import React, { useState, useEffect } from 'react';
import {
  CalendarDays, CheckCircle2, ChevronRight, Activity, HeartPulse,
  Clock, Award, ShieldCheck, Sparkles, XCircle, Pill, AlertTriangle
} from 'lucide-react';
import { getApiBase } from '../../apiConfig';

const API_BASE = getApiBase();

export function PatientRecoveryJourneyCard({ patientId, session, lang = 'en', onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const getStoredToken = () => {
      try {
        const stored = JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}');
        return session?.token || session?.access_token || stored?.token || stored?.access_token || '';
      } catch {
        return session?.token || session?.access_token || '';
      }
    };
    const token = getStoredToken();
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

  const hasStarted = Boolean(data.has_started) && Boolean(data.medication_start_date);
  const currentDay = hasStarted && data.current_day != null && Number(data.current_day) > 0
    ? Number(data.current_day)
    : 0;

  const tier = (data.deviation_tier || 'normal').toLowerCase();
  let badgeKey = 'on_track';
  let badgeClass = hasStarted ? 'badge-green' : 'badge-blue';

  if (!hasStarted) {
    badgeKey = 'awaiting_start';
    badgeClass = 'badge-blue';
  } else if (tier.includes('critical')) {
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
      critical: '● Healing Trajectory: Action Recommended',
      awaiting_start: '● Recovery Journey: Awaiting Medication Start'
    },
    ta: {
      on_track: '● குணமடைதல் முன்னேற்றம்: சரியான பாதையில்',
      mild: '● குணமடைதல் முன்னேற்றம்: லேசான விலகல்',
      moderate: '● குணமடைதல் முன்னேற்றம்: மிதமான விலகல்',
      critical: '● குணமடைதல் முன்னேற்றம்: தீவிர விலகல்',
      awaiting_start: '● மீட்புப் பயணம்: மருந்துகள் தொடக்கம் நிலுவையில்'
    },
    hi: {
      on_track: '● रिकवरी प्रगति: सही राह पर',
      mild: '● रिकवरी प्रगति: हल्का विचलन',
      moderate: '● रिकवरी प्रगति: मध्यम विचलन',
      critical: '● रिकवरी प्रगति: तत्काल ध्यान आवश्यक',
      awaiting_start: '● रिकवरी यात्रा: दवा शुरू होने की प्रतीक्षा'
    }
  };

  const contentMap = {
    en: {
      badge: (badgeLabels.en[badgeKey] || badgeLabels.en.on_track),
      pathway: 'Standard Clinical Recovery Pathway',
      journeyTitle: hasStarted
        ? `My Recovery Journey — Currently on Day ${currentDay}`
        : 'My Recovery Journey — Awaiting Medication Schedule',
      awaitingDesc: 'Your personalized daily recovery timeline begins on Day 1 once prescribed medications are scheduled by your attending care team.',
      checkinBtn: 'Complete Daily Check-in',
      checkinDoneToday: 'Today’s Check-in Complete',
      todayLabel: 'TODAY',
      pastLabel: 'RECORDED',
      futureLabel: 'UPCOMING',
      phaseLabel: 'Current Status',
      checkinsMetric: 'Check-ins Logged',
      adherenceMetric: 'Med Adherence',
      nextEventMetric: 'Next Follow-up',
      checkinDone: 'Check-in done',
      checkinPending: 'Pending today',
      checkinMissed: 'Check-in not logged',
      medsAllTaken: 'All meds taken',
      medsPartial: 'Partially done',
      medsNotLogged: 'Meds not logged',
      medsPendingToday: 'Pending today',
      medsNoneScheduled: 'None scheduled',
      projectedRecovery: 'Projected recovery',
      timelineSection: 'Longitudinal Daily Recovery Progress'
    },
    ta: {
      badge: (badgeLabels.ta[badgeKey] || badgeLabels.ta.on_track),
      pathway: 'நிலையான மருத்துவ மீட்புப் பாதை',
      journeyTitle: hasStarted
        ? `எனது மீட்புப் பயணம் — தற்போது நாள் ${currentDay}`
        : 'எனது மீட்புப் பயணம் — மருந்துகள் அட்டவணை நிலுவையில்',
      awaitingDesc: 'மருத்துவர் மருந்து அட்டவணையை பதிவு செய்தவுடன் உங்களின் தனிப்பயனாக்கப்பட்ட மீட்பு பயணம் நாள் 1-ல் தொடங்கும்.',
      checkinBtn: 'தினசரி சோதனையை முடிக்கவும்',
      checkinDoneToday: 'இன்றைய சோதனை முடிந்தது',
      todayLabel: 'இன்று',
      pastLabel: 'பதிவானது',
      futureLabel: 'வரவிருக்கும்',
      phaseLabel: 'தற்போதைய நிலை',
      checkinsMetric: 'பதிவான சோதனைகள்',
      adherenceMetric: 'மருந்து கடைபிடிப்பு',
      nextEventMetric: 'அடுத்த நிகழ்வு',
      checkinDone: 'சோதனை முடிந்தது',
      checkinPending: 'இன்று நிலுவையில்',
      checkinMissed: 'சோதனை பதிவு செய்யப்படவில்லை',
      medsAllTaken: 'அனைத்து மருந்துகளும் எடுக்கப்பட்டன',
      medsPartial: 'பகுதி முடிந்தது',
      medsNotLogged: 'மருந்துகள் எடுக்கப்படவில்லை',
      medsPendingToday: 'இன்று நிலுவை',
      medsNoneScheduled: 'மருந்துகள் திட்டமிடப்படவில்லை',
      projectedRecovery: 'எதிர்பார்க்கப்படும் மீட்பு',
      timelineSection: 'நீண்டகால தினசரி மீட்பு முன்னேற்றம்'
    },
    hi: {
      badge: (badgeLabels.hi[badgeKey] || badgeLabels.hi.on_track),
      pathway: 'मानक क्लिनिकल रिकवरी मार्ग',
      journeyTitle: hasStarted
        ? `मेरी रिकवरी यात्रा — वर्तमान में दिन ${currentDay}`
        : 'मेरी रिकवरी यात्रा — दवा अनुसूची की प्रतीक्षा',
      awaitingDesc: 'चिकित्सक द्वारा दवा अनुसूची निर्धारित किए जाने पर आपकी व्यक्तिगत दैनिक रिकवरी यात्रा दिन 1 से शुरू होगी।',
      checkinBtn: 'दैनिक चेक-इन पूरा करें',
      checkinDoneToday: 'आज का चेक-इन पूर्ण',
      todayLabel: 'आज',
      pastLabel: 'दर्ज',
      futureLabel: 'आगामी',
      phaseLabel: 'वर्तमान स्थिति',
      checkinsMetric: 'चेक-इन दर्ज',
      adherenceMetric: 'दवा पालन',
      nextEventMetric: 'अगला परामर्श',
      checkinDone: 'चेक-इन पूर्ण',
      checkinPending: 'आज प्रतीक्षारत',
      checkinMissed: 'चेक-इन दर्ज नहीं हुआ',
      medsAllTaken: 'सभी दवाइयां ली गईं',
      medsPartial: 'आंशिक रूप से ली गई',
      medsNotLogged: 'दवा दर्ज नहीं हुई',
      medsPendingToday: 'आज प्रतीक्षारत',
      medsNoneScheduled: 'कोई दवा नहीं',
      projectedRecovery: 'अनुमानित रिकवरी',
      timelineSection: 'दैनिक रिकवरी प्रगति'
    }
  };

  const t = contentMap[lang] || contentMap.en;
  const timeline = data.timeline || [];

  const todayItem = timeline.find(item => item.is_today);
  const isCheckedInToday = todayItem?.checkin_status === 'completed';

  return (
    <div className="card" style={{ padding: 22, marginTop: 18, background: 'var(--surface)', border: '1px solid var(--border)' }}>
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 46,
            height: 46,
            borderRadius: 12,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: hasStarted
              ? 'linear-gradient(135deg, rgba(0,212,184,0.18), rgba(56,189,248,0.18))'
              : 'rgba(148, 163, 184, 0.12)',
            color: hasStarted ? 'var(--p)' : 'var(--text-muted)'
          }}>
            <HeartPulse size={26} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`badge ${badgeClass}`} style={{ fontSize: 11, fontWeight: 700 }}>
                {t.badge}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {t.pathway} {hasStarted && data.medication_start_date ? `· Started ${data.medication_start_date}` : ''}
              </span>
            </div>
            <h3 style={{ margin: '4px 0 0', fontSize: 17, fontWeight: 800, color: 'var(--text)' }}>
              {t.journeyTitle}
            </h3>
          </div>
        </div>

        {hasStarted ? (
          isCheckedInToday ? (
            <button
              className="btn btn-secondary btn-sm"
              disabled
              style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, opacity: 0.9 }}
            >
              <CheckCircle2 size={13} style={{ color: 'var(--success)' }} /> {t.checkinDoneToday}
            </button>
          ) : (
            <button
              className="btn btn-primary btn-sm"
              onClick={() => onNavigate ? onNavigate('#/patient/check-in') : (window.location.hash = '#/patient/check-in')}
              style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <CheckCircle2 size={13} /> {t.checkinBtn}
            </button>
          )
        ) : null}
      </div>

      {/* When awaiting medication start */}
      {!hasStarted ? (
        <div style={{
          marginTop: 16,
          padding: '16px 20px',
          borderRadius: 10,
          background: 'var(--surface2)',
          border: '1px dashed var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 14
        }}>
          <Clock size={24} style={{ color: 'var(--p)', flexShrink: 0 }} />
          <div style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.5 }}>
            {t.awaitingDesc}
          </div>
        </div>
      ) : (
        <>
          {/* Key Clinical Telemetry Metrics Row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
            gap: 10,
            marginTop: 16,
            marginBottom: 16
          }}>
            <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                {t.phaseLabel}
              </div>
              <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--p)', marginTop: 2 }}>
                Day {currentDay} Active
              </div>
            </div>

            <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                {t.checkinsMetric}
              </div>
              <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text)', marginTop: 2 }}>
                {data.completed_checkins_count || 0} / {currentDay} Days
              </div>
            </div>

            <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                {t.adherenceMetric}
              </div>
              <div style={{ fontSize: 15, fontWeight: 800, color: data.adherence_rate >= 75 ? 'var(--success)' : 'var(--warn)', marginTop: 2 }}>
                {data.adherence_rate ?? 0}% <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-muted)' }}>({data.taken_doses || 0}/{data.scheduled_doses || 0})</span>
              </div>
            </div>

            <div style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                {t.nextEventMetric}
              </div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {data.next_event_summary || 'Rest & Monitoring'}
              </div>
            </div>
          </div>

          {/* Adaptive Daily Recovery Timeline */}
          <div style={{ marginTop: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                {t.timelineSection}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                Scroll horizontal to inspect full journey →
              </span>
            </div>

            <div style={{
              display: 'flex',
              gap: 12,
              overflowX: 'auto',
              paddingBottom: 8,
              paddingTop: 4,
              scrollbarWidth: 'thin'
            }}>
              {timeline.map((item) => {
                const isToday = item.is_today;
                const isPast = item.is_past;
                const isFuture = item.is_future;

                return (
                  <div
                    key={item.day_number}
                    style={{
                      flex: '0 0 200px',
                      padding: 12,
                      borderRadius: 10,
                      background: isToday
                        ? 'rgba(0, 212, 184, 0.08)'
                        : isFuture
                          ? 'rgba(15, 23, 42, 0.35)'
                          : 'var(--surface2)',
                      border: isToday
                        ? '1.5px solid var(--p)'
                        : isFuture
                          ? '1px dashed var(--border)'
                          : '1px solid var(--border)',
                      boxShadow: isToday ? '0 0 12px rgba(0, 212, 184, 0.2)' : 'none',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      gap: 8
                    }}
                  >
                    {/* Top day header */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{
                          fontSize: 13,
                          fontWeight: 800,
                          color: isToday ? 'var(--p)' : 'var(--text)'
                        }}>
                          Day {item.day_number}
                        </span>
                        {isToday ? (
                          <span className="badge badge-teal" style={{ fontSize: 9, fontWeight: 800 }}>
                            {t.todayLabel}
                          </span>
                        ) : isPast ? (
                          <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>
                            {item.formatted_date}
                          </span>
                        ) : (
                          <span className="badge badge-blue" style={{ fontSize: 9 }}>
                            {item.formatted_date || t.futureLabel}
                          </span>
                        )}
                      </div>

                      {/* Daily Check-in status line */}
                      {!isFuture && (
                        <div style={{ fontSize: 11.5, display: 'flex', alignItems: 'center', gap: 6, margin: '4px 0' }}>
                          {item.checkin_status === 'completed' ? (
                            <span style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}>
                              <CheckCircle2 size={12} /> {t.checkinDone} (Pain {item.checkin_pain}/10)
                            </span>
                          ) : isToday ? (
                            <span style={{ color: 'var(--p)', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}>
                              <Clock size={12} /> {t.checkinPending}
                            </span>
                          ) : (
                            <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
                              <XCircle size={12} /> {t.checkinMissed}
                            </span>
                          )}
                        </div>
                      )}

                      {/* Medication status line */}
                      {!isFuture && (
                        <div style={{ fontSize: 11.5, display: 'flex', alignItems: 'center', gap: 6, margin: '4px 0' }}>
                          {item.med_status === 'all_taken' ? (
                            <span style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}>
                              <Pill size={12} /> {t.medsAllTaken} ({item.med_taken}/{item.med_scheduled})
                            </span>
                          ) : item.med_status === 'partially_taken' ? (
                            <span style={{ color: 'var(--warn)', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}>
                              <Pill size={12} /> {t.medsPartial} ({item.med_taken}/{item.med_scheduled})
                            </span>
                          ) : item.med_status === 'not_taken' ? (
                            <span style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: 4 }}>
                              <Pill size={12} /> {isToday ? t.medsPendingToday : t.medsNotLogged}
                            </span>
                          ) : (
                            <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
                              <Pill size={12} /> {t.medsNoneScheduled}
                            </span>
                          )}
                        </div>
                      )}

                      {/* Future schedules & genuine likelihood predictions */}
                      {isFuture && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, margin: '4px 0' }}>
                          {item.has_appointment ? (
                            <span style={{ color: '#38bdf8', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                              <CalendarDays size={12} /> {item.appointment_detail}
                            </span>
                          ) : item.projected_score != null ? (
                            <span style={{ color: 'var(--p)', fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4 }}>
                              <Activity size={12} /> {t.projectedRecovery}: {item.projected_score} pts
                            </span>
                          ) : null}

                          <span style={{ color: 'var(--text-dim)', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                            <Pill size={12} /> {item.med_message}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Small status message under the day */}
                    <div style={{
                      fontSize: 10.5,
                      color: 'var(--text-dim)',
                      lineHeight: 1.35,
                      borderTop: '1px solid var(--border)',
                      paddingTop: 6,
                      fontStyle: isFuture ? 'italic' : 'normal'
                    }}>
                      {item.summary_message}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
