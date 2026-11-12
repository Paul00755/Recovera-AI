import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, ShieldCheck, AlertTriangle, AlertCircle, RefreshCw,
  TrendingUp, TrendingDown, Info, CheckCircle2, ChevronRight,
  Clock, Calendar, FileText, Check, X, ShieldAlert, Cpu, Sparkles
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function DigitalTwinTab({ patientId, session }) {
  const [loading, setLoading] = useState(true);
  const [recomputing, setRecomputing] = useState(false);
  const [stateData, setStateData] = useState(null);
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selectedDay, setSelectedDay] = useState(null);
  const [reviewNote, setReviewNote] = useState('');
  const [reviewingAlertId, setReviewingAlertId] = useState(null);
  const [actionSuccess, setActionSuccess] = useState(null);
  const [error, setError] = useState(null);

  const token = session?.token || JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}')?.token;
  const headers = useMemo(() => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }), [token]);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [sRes, tRes, aRes] = await Promise.all([
        fetch(`${API_BASE}/api/digital-twin/patients/${patientId}/state`, { headers }),
        fetch(`${API_BASE}/api/digital-twin/patients/${patientId}/trajectory`, { headers }),
        fetch(`${API_BASE}/api/digital-twin/patients/${patientId}/alerts`, { headers })
      ]);

      if (!sRes.ok || !tRes.ok) {
        throw new Error('Failed to retrieve Digital Twin state data');
      }

      const sData = await sRes.json();
      const tData = await tRes.json();
      const aData = aRes.ok ? await aRes.json() : [];

      setStateData(sData);
      setTrajectoryData(tData);
      setAlerts(aData);
      setSelectedDay(tData.current_day || 1);
    } catch (err) {
      console.error('Digital Twin error:', err);
      setError(err.message || 'Error communicating with Digital Twin engine');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (patientId) {
      fetchData();
    }
  }, [patientId]);

  const handleRecompute = async () => {
    try {
      setRecomputing(true);
      const res = await fetch(`${API_BASE}/api/digital-twin/patients/${patientId}/recompute`, {
        method: 'POST',
        headers
      });
      if (!res.ok) throw new Error('Recompute failed');
      await fetchData();
      setActionSuccess('Digital Twin state successfully recalculated.');
      setTimeout(() => setActionSuccess(null), 3500);
    } catch (err) {
      alert(`Error recomputing digital twin: ${err.message}`);
    } finally {
      setRecomputing(false);
    }
  };

  const handleReviewAlert = async (alertId, action) => {
    try {
      setReviewingAlertId(alertId);
      const res = await fetch(`${API_BASE}/api/digital-twin/alerts/${alertId}/review`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          review_action: action,
          review_notes: reviewNote || `Clinician set status to ${action}`
        })
      });
      if (!res.ok) throw new Error('Review submission failed');
      setReviewNote('');
      await fetchData();
      setActionSuccess(`Alert marked as ${action}.`);
      setTimeout(() => setActionSuccess(null), 3500);
    } catch (err) {
      alert(`Failed to review alert: ${err.message}`);
    } finally {
      setReviewingAlertId(null);
    }
  };

  if (loading && !stateData) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
        <RefreshCw size={28} className="animate-spin" style={{ margin: '0 auto 12px', display: 'block', color: 'var(--p)' }} />
        <div style={{ fontWeight: 600 }}>Loading Patient Recovery Digital Twin...</div>
        <div style={{ fontSize: 13, marginTop: 4 }}>Synthesizing longitudinal corridor & deep learning forecasts</div>
      </div>
    );
  }

  if (error && !stateData) {
    return (
      <div className="card" style={{ padding: 24, textAlign: 'center', borderColor: 'var(--danger)' }}>
        <AlertTriangle size={32} style={{ color: 'var(--danger)', margin: '0 auto 12px' }} />
        <h3 style={{ margin: '0 0 8px', color: 'var(--text)' }}>Digital Twin Unavailable</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>{error}</p>
        <button className="btn btn-primary btn-sm" onClick={fetchData} style={{ marginTop: 12 }}>
          Retry Initialization
        </button>
      </div>
    );
  }

  const tier = stateData?.deviation_tier || 'normal';
  const tierConfig = {
    normal: { label: 'Optimal Recovery Progression', badge: 'badge-green', color: 'var(--success)', desc: 'Telemetry strictly aligns with expected post-discharge recovery corridor.' },
    mild_deviation: { label: 'Mild Healing Adjustment', badge: 'badge-amber', color: 'var(--warn)', desc: 'Minor variance in symptom progression or unlogged telemetry; care oversight recommended.' },
    moderate_deviation: { label: 'Moderate Trajectory Deviation', badge: 'badge-amber', color: '#f97316', desc: 'Sustained divergence from clinical milestones; clinician review recommended.' },
    critical_disruption: { label: 'Critical Disruption Flag', badge: 'badge-red', color: 'var(--danger)', desc: 'Severe divergence detected across acute vitals and symptom trajectories.' }
  }[tier] || { label: 'Active Monitoring', badge: 'badge-teal', color: 'var(--p)', desc: 'Standard recovery tracking' };

  const components = stateData?.component_scores || {};
  const corridor = trajectoryData?.corridor_milestones || [];
  const observed = trajectoryData?.observed_points || [];
  const forecasts = trajectoryData?.forecasts || [];

  // SVG Chart Dimensions
  const chartW = 760;
  const chartH = 220;
  const padL = 40;
  const padR = 24;
  const padT = 20;
  const padB = 34;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;

  const getX = (day) => padL + ((day - 1) / 13) * plotW;
  const getY = (score) => padT + (1 - (score / 100)) * plotH;

  // Build corridor polygon (envelope [min, max])
  const corridorPoints = [];
  corridor.forEach((m) => {
    corridorPoints.push(`${getX(m.day_number)},${getY(m.corridor_score_max)}`);
  });
  for (let i = corridor.length - 1; i >= 0; i--) {
    corridorPoints.push(`${getX(corridor[i].day_number)},${getY(corridor[i].corridor_score_min)}`);
  }
  const corridorSvgPath = corridorPoints.join(' ');

  // Target trajectory centerline
  const targetSvgPoints = corridor.map(m => `${getX(m.day_number)},${getY(m.target_recovery_score)}`).join(' ');

  // Observed points
  const observedWithCoords = observed.map((pt) => {
    const score = pt.submitted ? Math.max(10, 100 - (pt.pain_level || 0) * 10) : null;
    return {
      ...pt,
      score,
      x: getX(pt.day),
      y: score !== null ? getY(score) : null
    };
  });
  const observedPointsWithScores = observedWithCoords.filter(pt => pt.score !== null);
  const observedLine = observedPointsWithScores.map(pt => `${pt.x},${pt.y}`).join(' ');

  // Forecast points & Anchor: Project smoothly from current day (Today)
  const currentDayNum = trajectoryData?.current_day || 1;
  const currentScore = stateData?.recovery_score || 80.0;
  const anchorPoint = {
    x: getX(currentDayNum),
    y: getY(currentScore)
  };

  const forecastPoints = [];
  if (forecasts.length > 0) {
    forecastPoints.push(`${anchorPoint.x},${anchorPoint.y}`);
    forecasts.forEach(f => {
      forecastPoints.push(`${getX(f.day_number)},${getY(f.point_forecast)}`);
    });
  }
  const forecastLine = forecastPoints.join(' ');

  // Selected Day Details
  const activeMilestone = corridor.find(m => m.day_number === selectedDay) || corridor[0];
  const activeObserved = observed.find(o => o.day === selectedDay);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {actionSuccess && (
        <div className="alert alert-success" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', borderRadius: 8 }}>
          <CheckCircle2 size={16} />
          <span>{actionSuccess}</span>
        </div>
      )}

      {/* Hero Overview Header */}
      <div className="card" style={{ padding: 22, background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--surface2)',
              color: 'var(--p)',
              border: '1px solid var(--border)'
            }}>
              <Cpu size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span className={`badge ${tierConfig.badge}`} style={{ fontSize: 11, fontWeight: 700 }}>
                  ● {tierConfig.label}
                </span>
                <span className="badge badge-neutral" style={{ fontSize: 11 }}>
                  Day {trajectoryData?.current_day || 1} of 14
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Evaluated: {stateData?.last_calculated_at ? new Date(stateData.last_calculated_at).toLocaleTimeString() : 'Just now'}
                </span>
              </div>
              <h2 style={{ margin: '4px 0 2px', fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>
                {stateData?.patient_name} — Recovery Digital Twin
              </h2>
              <p style={{ margin: 0, fontSize: 13, color: 'var(--text-dim)', maxWidth: 650 }}>
                {tierConfig.desc}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              className="btn btn-ghost btn-sm"
              onClick={handleRecompute}
              disabled={recomputing}
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <RefreshCw size={14} className={recomputing ? 'animate-spin' : ''} />
              {recomputing ? 'Recalculating...' : 'Re-evaluate State'}
            </button>
          </div>
        </div>

        {/* Top Metric Strip */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginTop: 20 }}>
          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Projected Recovery Score
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: tierConfig.color, fontVariantNumeric: 'tabular-nums' }}>
                {stateData?.recovery_score}%
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>/ 100</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              R_t = 100 × (1 − D_t)
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Composite Deviation (D_t)
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
                {stateData?.deviation_score}%
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>divergence</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Threshold: {tierConfig.label}
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              24h Trajectory Forecast
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: '#38bdf8', fontVariantNumeric: 'tabular-nums' }}>
                {forecasts[0]?.point_forecast ? `${forecasts[0].point_forecast}%` : '—'}
              </span>
              {forecasts[0] && (
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                  ± {(forecasts[0].confidence_upper_95 - forecasts[0].point_forecast).toFixed(1)}%
                </span>
              )}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              {forecasts[0]?.model_architecture || 'Corridor Projection'} (95% CI)
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Pending Deviation Flags
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: alerts.length > 0 ? 'var(--warn)' : 'var(--success)', fontVariantNumeric: 'tabular-nums' }}>
                {alerts.length}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>unreviewed</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              {alerts.length > 0 ? 'Clinician review required' : 'No active deviations'}
            </div>
          </div>
        </div>
      </div>

      {/* 5-Factor Explainable Deviation Breakdown Cards */}
      <div className="card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: 'var(--text)' }}>
              Explainable Deviation Breakdown Vector
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--text-muted)' }}>
              Mathematical formulation: D_t = 0.35·S_t + 0.25·A_t + 0.20·T_t + 0.10·F_t + 0.10·Q_t
            </p>
          </div>
          <span className="badge badge-teal" style={{ fontSize: 11 }}>Deterministic Weights</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
          {[
            {
              symbol: 'S_t',
              label: 'Symptom Divergence',
              weight: '35%',
              score: components.symptom_deviation_st || 0,
              weighted: ((components.symptom_deviation_st || 0) * 0.35).toFixed(1),
              tooltip: 'Divergence of pain and reported severity from expected milestone envelope'
            },
            {
              symbol: 'A_t',
              label: 'Medication Adherence Gap',
              weight: '25%',
              score: components.adherence_penalty_at || 0,
              weighted: ((components.adherence_penalty_at || 0) * 0.25).toFixed(1),
              tooltip: 'Unrecorded or delayed scheduled medication occurrences over 14-day history'
            },
            {
              symbol: 'T_t',
              label: 'Negative Trend Slope',
              weight: '20%',
              score: components.consistency_penalty_tt || 0,
              weighted: ((components.consistency_penalty_tt || 0) * 0.20).toFixed(1),
              tooltip: 'Empirical slope of worsening pain or deteriorating daily trajectory'
            },
            {
              symbol: 'F_t',
              label: 'Acute Vital / Follow-up Flag',
              weight: '10%',
              score: components.fever_vital_flag_ft || 0,
              weighted: ((components.fever_vital_flag_ft || 0) * 0.10).toFixed(1),
              tooltip: 'Binary flag for acute vital disruption (fever/respiratory) and missed clinical appointments'
            },
            {
              symbol: 'Q_t',
              label: 'Telemetry Missingness',
              weight: '10%',
              score: components.sentiment_penalty_qt || 0,
              weighted: ((components.sentiment_penalty_qt || 0) * 0.10).toFixed(1),
              tooltip: 'Penalizes irregular gaps and missing daily health check-in submissions'
            }
          ].map((c) => (
            <div key={c.symbol} style={{
              padding: 12,
              borderRadius: 8,
              background: 'var(--surface2)',
              border: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 800, fontSize: 13, color: 'var(--p)' }}>{c.symbol}</span>
                  <span style={{ fontSize: 10, color: 'var(--text-dim)', background: 'var(--surface3)', padding: '2px 5px', borderRadius: 4 }}>
                    {c.weight}
                  </span>
                </div>
                <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text)', marginTop: 4 }}>
                  {c.label}
                </div>
              </div>
              <div style={{ marginTop: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                  <span>Raw: {c.score}%</span>
                  <span style={{ fontWeight: 700, color: 'var(--text)' }}>+{c.weighted}%</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: 'var(--surface3)', marginTop: 4, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(100, c.score)}%`,
                    height: '100%',
                    background: c.score > 50 ? 'var(--warn)' : 'var(--p)'
                  }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive 14-Day Trajectory Corridor & Deep Learning Forecast */}
      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>
              14-Day Post-Discharge Clinical Corridor & Forecast
            </h3>
            <p style={{ margin: '3px 0 0', fontSize: 12.5, color: 'var(--text-muted)' }}>
              Green shaded envelope indicates the clinician-approved recovery corridor. The dashed line depicts next 48h projections with 95% confidence intervals.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 16, fontSize: 11.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 14, height: 10, background: 'rgba(16, 185, 129, 0.25)', border: '1px dashed #10b981', borderRadius: 2 }} />
              <span>Recovery Corridor [Y_min, Y_max]</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 14, height: 3, background: 'var(--p)', borderRadius: 2 }} />
              <span>Observed Patient Telemetry</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 14, height: 2, borderTop: '2px dashed #38bdf8' }} />
              <span>Deep Learning Forecast (±95% CI)</span>
            </div>
          </div>
        </div>

        {/* SVG Visualization */}
        <div style={{ overflowX: 'auto', paddingBottom: 10 }}>
          <svg viewBox={`0 0 ${chartW} ${chartH}`} style={{ width: '100%', height: 'auto', minWidth: 600 }}>
            <defs>
              <linearGradient id="corridorGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.08" />
              </linearGradient>
              <linearGradient id="forecastGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.1" />
              </linearGradient>
            </defs>

            {/* Grid Lines */}
            {[20, 40, 60, 80, 100].map(s => (
              <g key={s}>
                <line
                  x1={padL}
                  y1={getY(s)}
                  x2={chartW - padR}
                  y2={getY(s)}
                  stroke="var(--border)"
                  strokeDasharray="3 3"
                  strokeWidth="0.8"
                />
                <text
                  x={padL - 8}
                  y={getY(s) + 4}
                  textAnchor="end"
                  fontSize="10"
                  fill="var(--text-dim)"
                >
                  {s}
                </text>
              </g>
            ))}

            {/* Shaded Corridor Envelope */}
            {corridor.length > 0 && (
              <polygon
                points={corridorSvgPath}
                fill="url(#corridorGrad)"
                stroke="#10b981"
                strokeWidth="1"
                strokeDasharray="4 4"
                strokeOpacity="0.7"
              />
            )}

            {/* Corridor Centerline */}
            {corridor.length > 0 && (
              <polyline
                points={targetSvgPoints}
                fill="none"
                stroke="#10b981"
                strokeWidth="1.5"
                strokeOpacity="0.8"
              />
            )}

            {/* Observed Patient Trajectory Line */}
            {observedLine && (
              <polyline
                points={observedLine}
                fill="none"
                stroke="var(--p)"
                strokeWidth="2.5"
              />
            )}

            {/* Forecast Line */}
            {forecastLine && (
              <polyline
                points={forecastLine}
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
                strokeDasharray="4 4"
              />
            )}

            {/* Forecast Uncertainty Envelope */}
            {forecasts.length > 0 && (
              <polygon
                points={`
                  ${anchorPoint.x},${anchorPoint.y}
                  ${getX(forecasts[0].day_number)},${getY(forecasts[0].confidence_upper_95)}
                  ${getX(forecasts[1]?.day_number || forecasts[0].day_number)},${getY(forecasts[1]?.confidence_upper_95 || forecasts[0].confidence_upper_95)}
                  ${getX(forecasts[1]?.day_number || forecasts[0].day_number)},${getY(forecasts[1]?.confidence_lower_95 || forecasts[0].confidence_lower_95)}
                  ${getX(forecasts[0].day_number)},${getY(forecasts[0].confidence_lower_95)}
                `}
                fill="url(#forecastGrad)"
                stroke="#38bdf8"
                strokeWidth="0.5"
                strokeOpacity="0.5"
              />
            )}

            {/* Day Markers on X-Axis & Interactive Targets */}
            {corridor.map((m) => {
              const x = getX(m.day_number);
              const isSelected = selectedDay === m.day_number;
              const isToday = m.day_number === (trajectoryData?.current_day || 1);
              const obs = observed.find(o => o.day === m.day_number);
              const hasCheckin = obs?.submitted;

              return (
                <g key={m.day_number} style={{ cursor: 'pointer' }} onClick={() => setSelectedDay(m.day_number)}>
                  {/* Vertical Hover Column Target */}
                  <rect
                    x={x - (plotW / 26)}
                    y={padT}
                    width={plotW / 13}
                    height={plotH}
                    fill={isSelected ? 'rgba(0, 212, 184, 0.12)' : (isToday ? 'rgba(56, 189, 248, 0.05)' : 'transparent')}
                  />

                  {/* Today subtle vertical dash indicator */}
                  {isToday && (
                    <line
                      x1={x}
                      y1={padT}
                      x2={x}
                      y2={chartH - padB}
                      stroke="#38bdf8"
                      strokeDasharray="2 3"
                      strokeWidth="1"
                      strokeOpacity="0.4"
                    />
                  )}

                  {/* X Axis Label */}
                  <text
                    x={x}
                    y={chartH - 12}
                    textAnchor="middle"
                    fontSize={isSelected ? "11" : "10"}
                    fontWeight={isSelected || isToday ? "700" : "500"}
                    fill={isSelected ? "var(--p)" : (isToday ? "#38bdf8" : "var(--text-dim)")}
                  >
                    D{m.day_number}
                  </text>
                  {isToday && (
                    <text
                      x={x}
                      y={chartH - 2}
                      textAnchor="middle"
                      fontSize="8"
                      fontWeight="700"
                      fill="#38bdf8"
                    >
                      NOW
                    </text>
                  )}

                  {/* Observed Data Circle */}
                  {hasCheckin && (
                    <circle
                      cx={x}
                      cy={getY(Math.max(10, 100 - (obs.pain_level || 0) * 10))}
                      r={isSelected ? 6 : 4.5}
                      fill="var(--p)"
                      stroke="#fff"
                      strokeWidth="1.5"
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Selected Day Inspector Card */}
        {activeMilestone && (
          <div style={{
            marginTop: 14,
            padding: '12px 16px',
            borderRadius: 8,
            background: 'var(--surface2)',
            border: '1px solid var(--border)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 12
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontWeight: 800, fontSize: 13, color: 'var(--text)' }}>
                  Day {activeMilestone.day_number} Milestone Inspector
                </span>
                {activeMilestone.target_date && (
                  <span style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>
                    ({new Date(activeMilestone.target_date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}{activeMilestone.day_number === (trajectoryData?.current_day || 1) ? ' — Today' : ''})
                  </span>
                )}
                <span className="badge badge-teal" style={{ fontSize: 10 }}>
                  Target Score: {activeMilestone.target_recovery_score}%
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Corridor: [{activeMilestone.corridor_score_min}% — {activeMilestone.corridor_score_max}%]
                </span>
              </div>
              <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-dim)' }}>
                {activeMilestone.milestone_description} (Expected pain: {activeMilestone.expected_pain_min}–{activeMilestone.expected_pain_max}/10, Activity: {activeMilestone.expected_activity || 'routine'})
              </p>
            </div>

            <div>
              {activeObserved?.submitted ? (
                <div style={{ textAlign: 'right', fontSize: 12 }}>
                  <span className="badge badge-green" style={{ fontSize: 11 }}>
                    Submitted Check-in
                  </span>
                  <div style={{ color: 'var(--text)', fontWeight: 600, marginTop: 2 }}>
                    Reported Pain: {activeObserved.pain_level}/10 ({activeObserved.pain_location || 'General'})
                  </div>
                </div>
              ) : activeObserved?.is_future ? (
                <div style={{ textAlign: 'right', fontSize: 12 }}>
                  <span className="badge badge-teal" style={{ fontSize: 11, background: 'rgba(56,189,248,0.1)', color: '#38bdf8', border: '1px solid rgba(56,189,248,0.3)' }}>
                    Future Milestone Target
                  </span>
                  <div style={{ color: 'var(--text-dim)', fontSize: 11, marginTop: 2 }}>
                    Scheduled for {activeMilestone.target_date ? new Date(activeMilestone.target_date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : `Day ${activeMilestone.day_number}`}
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'right', fontSize: 12 }}>
                  <span className="badge badge-amber" style={{ fontSize: 11 }}>
                    No Telemetry Recorded
                  </span>
                  <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
                    Elapsed Recovery Window
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Clinician Review Queue for Deviation Alerts */}
      <div className="card" style={{ padding: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: 'var(--text)' }}>
              Clinician-in-the-Loop Deviation Review Queue
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--text-muted)' }}>
              All algorithmic deviation triggers require human clinician review before care escalation.
            </p>
          </div>
          <span className="badge badge-teal" style={{ fontSize: 11 }}>
            Safety Gate
          </span>
        </div>

        {alerts.length === 0 ? (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)', background: 'var(--surface2)', borderRadius: 8 }}>
            <ShieldCheck size={28} style={{ color: 'var(--success)', margin: '0 auto 8px', display: 'block' }} />
            <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--text)' }}>
              No Unreviewed Deviation Flags
            </div>
            <div style={{ fontSize: 12, marginTop: 2 }}>
              Patient telemetry is currently within supervised tolerance boundaries.
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {alerts.map((alert) => (
              <div key={alert.id} style={{
                padding: 14,
                borderRadius: 8,
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                display: 'flex',
                flexDirection: 'column',
                gap: 10
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className={`badge ${alert.severity === 'severe' ? 'badge-red' : 'badge-amber'}`} style={{ fontSize: 10, textTransform: 'uppercase' }}>
                        {alert.severity}
                      </span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
                        Deviation Score: {alert.deviation_score != null ? `${alert.deviation_score}%` : 'Flagged'}
                      </span>
                      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        Detected: {alert.created_at ? new Date(alert.created_at).toLocaleString() : 'Recent'}
                      </span>
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text)', marginTop: 4 }}>
                      {alert.trigger_reason}
                    </div>
                  </div>

                  <span className="badge badge-teal" style={{ fontSize: 11 }}>
                    Status: {alert.status.toUpperCase()}
                  </span>
                </div>

                {alert.status === 'pending' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                    <input
                      type="text"
                      className="input"
                      placeholder="Add clinician note or clinical rationale..."
                      value={reviewNote}
                      onChange={(e) => setReviewNote(e.target.value)}
                      style={{ flex: 1, minWidth: 200, fontSize: 12 }}
                    />
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => handleReviewAlert(alert.id, 'acknowledged')}
                      disabled={reviewingAlertId === alert.id}
                      style={{ fontSize: 11.5 }}
                    >
                      <Check size={12} /> Acknowledge
                    </button>
                    <button
                      className="btn btn-ghost btn-sm"
                      onClick={() => handleReviewAlert(alert.id, 'dismissed')}
                      disabled={reviewingAlertId === alert.id}
                      style={{ fontSize: 11.5 }}
                    >
                      <X size={12} /> Dismiss
                    </button>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() => handleReviewAlert(alert.id, 'escalated')}
                      disabled={reviewingAlertId === alert.id}
                      style={{ fontSize: 11.5 }}
                    >
                      <AlertTriangle size={12} /> Escalate to MD
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Mandatory Decision-Support Boundary Notice */}
      <div style={{
        padding: '12px 16px',
        borderRadius: 8,
        background: 'var(--surface2)',
        border: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 12
      }}>
        <Info size={18} style={{ color: 'var(--p)', flexShrink: 0 }} />
        <div style={{ fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.45 }}>
          <strong style={{ color: 'var(--text)' }}>Clinical Decision Support System:</strong> The Recovera AI Patient Recovery Digital Twin provides explainable deterministic and statistical monitoring for post-discharge recovery trajectories. It does not formulate autonomous clinical diagnoses, alter prescribed medication protocols, or override attending physician discretion.
        </div>
      </div>
    </div>
  );
}
