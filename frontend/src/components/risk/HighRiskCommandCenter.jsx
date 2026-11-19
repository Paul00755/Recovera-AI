import React, { useState, useEffect, useMemo, Fragment } from 'react';
import {
  AlertTriangle, ShieldAlert, CheckCircle2, RefreshCw, Search,
  Filter, TrendingUp, TrendingDown, Clock, Calendar, User,
  ChevronRight, AlertCircle, HelpCircle, Activity, HeartPulse,
  Eye, Edit3, X, Check, ShieldCheck, ChevronDown, ChevronUp,
  Sparkles, MessageSquare, ClipboardList, Pill
} from 'lucide-react';
import { getApiBase } from '../../apiConfig';

const API_BASE = getApiBase();

export function HighRiskCommandCenter({ session, navigate }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState('all'); // all, red, yellow, repeated_missed, worsening, insufficient
  const [searchTerm, setSearchTerm] = useState('');
  const [overrideModal, setOverrideModal] = useState(null); // patient obj or null
  const [overrideLevel, setOverrideLevel] = useState('green');
  const [overrideReason, setOverrideReason] = useState('');
  const [actionSuccess, setActionSuccess] = useState(null);
  const [error, setError] = useState(null);
  const [expandedPatientId, setExpandedPatientId] = useState(null);

  const getStoredToken = () => {
    try {
      const stored = JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}');
      return session?.token || session?.access_token || stored?.token || stored?.access_token || '';
    } catch {
      return session?.token || session?.access_token || '';
    }
  };
  const token = getStoredToken();
  const headers = useMemo(() => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }), [token]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/api/patients/high-risk-overview`, { headers });
      if (!res.ok) throw new Error('Failed to retrieve clinical risk overview');
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error('High Risk Overview Error:', err);
      setError(err.message || 'Error communicating with risk identification engine');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleReevaluate = async (patientId) => {
    try {
      const res = await fetch(`${API_BASE}/api/patients/${patientId}/risk/evaluate`, {
        method: 'POST',
        headers
      });
      if (!res.ok) throw new Error('Re-evaluation failed');
      await loadData();
      setActionSuccess('Clinical risk score re-evaluated successfully.');
      setTimeout(() => setActionSuccess(null), 3500);
    } catch (err) {
      alert(`Re-evaluation error: ${err.message}`);
    }
  };

  const handleOverrideSubmit = async (e) => {
    e.preventDefault();
    if (!overrideModal) return;
    if (!overrideReason || overrideReason.trim().length < 5) {
      alert('Please document a valid clinical rationale for the override (minimum 5 characters).');
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/patients/${overrideModal.patient_id}/risk/override`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          level: overrideLevel,
          override_reason: overrideReason
        })
      });
      if (!res.ok) throw new Error('Failed to submit clinician override');
      setOverrideModal(null);
      setOverrideReason('');
      await loadData();
      setActionSuccess('Clinician override applied and recorded in audit log.');
      setTimeout(() => setActionSuccess(null), 3500);
    } catch (err) {
      alert(`Override error: ${err.message}`);
    }
  };

  const filteredPatients = useMemo(() => {
    if (!data?.patients) return [];
    let list = data.patients;

    if (filter === 'red') {
      list = list.filter(p => p.risk_level === 'red');
    } else if (filter === 'yellow') {
      list = list.filter(p => p.risk_level === 'yellow');
    } else if (filter === 'repeated_missed') {
      list = list.filter(p => p.has_repeated_missed_meds);
    } else if (filter === 'worsening') {
      list = list.filter(p => p.has_worsening_symptoms);
    } else if (filter === 'insufficient') {
      list = list.filter(p => p.risk_level === 'insufficient_data');
    }

    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      list = list.filter(p =>
        p.name?.toLowerCase().includes(q) ||
        p.patient_code?.toLowerCase().includes(q) ||
        p.assigned_doctor?.toLowerCase().includes(q)
      );
    }

    return list;
  }, [data, filter, searchTerm]);

  const summary = data?.summary || {
    total_active_patients: 0,
    green_count: 0,
    yellow_count: 0,
    red_count: 0,
    needing_review_count: 0,
    repeated_missed_meds_count: 0,
    worsening_symptoms_count: 0,
    upcoming_missed_followups_count: 0
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {actionSuccess && (
        <div className="alert alert-success" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderRadius: 8 }}>
          <CheckCircle2 size={16} />
          <span>{actionSuccess}</span>
        </div>
      )}

      {error && (
        <div className="alert alert-danger" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderRadius: 8 }}>
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Summary Counters Bar */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10 }}>
        {/* Total Active */}
        <div className="card" style={{ padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Total Active</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--text)', marginTop: 2 }}>{summary.total_active_patients}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Patients on roster</div>
        </div>

        {/* Red (High Concern) */}
        <div
          className="card"
          onClick={() => setFilter('red')}
          style={{
            padding: '12px 14px',
            background: filter === 'red' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(239, 68, 68, 0.06)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            cursor: 'pointer'
          }}
        >
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: '#ef4444', fontWeight: 700 }}>Red (High)</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: '#ef4444', marginTop: 2 }}>{summary.red_count}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Immediate attention</div>
        </div>

        {/* Yellow (Moderate Concern) */}
        <div
          className="card"
          onClick={() => setFilter('yellow')}
          style={{
            padding: '12px 14px',
            background: filter === 'yellow' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(245, 158, 11, 0.06)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            cursor: 'pointer'
          }}
        >
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: '#f59e0b', fontWeight: 700 }}>Yellow (Moderate)</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: '#f59e0b', marginTop: 2 }}>{summary.yellow_count}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Monitoring required</div>
        </div>

        {/* Green (Low Concern) */}
        <div
          className="card"
          onClick={() => setFilter('all')}
          style={{ padding: '12px 14px', background: 'rgba(34, 197, 94, 0.06)', border: '1px solid rgba(34, 197, 94, 0.3)', cursor: 'pointer' }}
        >
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: '#22c55e', fontWeight: 700 }}>Green (Low)</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: '#22c55e', marginTop: 2 }}>{summary.green_count}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Optimal corridor</div>
        </div>

        {/* Needing Clinical Review */}
        <div className="card" style={{ padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Need Review</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--text)', marginTop: 2 }}>{summary.needing_review_count}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Red + Yellow active</div>
        </div>

        {/* Repeated Missed Meds */}
        <div
          className="card"
          onClick={() => setFilter('repeated_missed')}
          style={{
            padding: '12px 14px',
            background: filter === 'repeated_missed' ? 'rgba(239, 68, 68, 0.15)' : 'var(--surface2)',
            border: '1px solid var(--border)',
            cursor: 'pointer'
          }}
        >
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Repeated Missed</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: summary.repeated_missed_meds_count > 0 ? '#ef4444' : 'var(--text)', marginTop: 2 }}>
            {summary.repeated_missed_meds_count}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Adherence hazard</div>
        </div>

        {/* Worsening Symptoms */}
        <div
          className="card"
          onClick={() => setFilter('worsening')}
          style={{
            padding: '12px 14px',
            background: filter === 'worsening' ? 'rgba(245, 158, 11, 0.15)' : 'var(--surface2)',
            border: '1px solid var(--border)',
            cursor: 'pointer'
          }}
        >
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Worsening Pain</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: summary.worsening_symptoms_count > 0 ? '#f59e0b' : 'var(--text)', marginTop: 2 }}>
            {summary.worsening_symptoms_count}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Pain/Fever flags</div>
        </div>

        {/* Upcoming / Missed Follow-ups */}
        <div className="card" style={{ padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10.5, textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>Follow-ups</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--text)', marginTop: 2 }}>{summary.upcoming_missed_followups_count}</div>
          <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Upcoming or missed</div>
        </div>
      </div>

      {/* Main Command Center Card */}
      <div className="card" style={{ padding: 22, background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="section-head" style={{ marginBottom: 16 }}>
          <div>
            <span className="eyebrow">CLINICAL TRIAGE HUB</span>
            <h2 style={{ margin: '2px 0 0', fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>
              High-Risk Patient Identification & Early Warning System
            </h2>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={loadData} title="Refresh risk classifications">
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
          </button>
        </div>

        {/* Filters and Search Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <button
              className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setFilter('all')}
              style={{ fontSize: 11.5 }}
            >
              All Active ({summary.total_active_patients})
            </button>
            <button
              className={`btn btn-sm ${filter === 'red' ? 'btn-danger' : 'btn-ghost'}`}
              onClick={() => setFilter('red')}
              style={{ fontSize: 11.5 }}
            >
              Red (High Concern) ({summary.red_count})
            </button>
            <button
              className={`btn btn-sm ${filter === 'yellow' ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setFilter('yellow')}
              style={{ fontSize: 11.5 }}
            >
              Yellow (Moderate) ({summary.yellow_count})
            </button>
            <button
              className={`btn btn-sm ${filter === 'repeated_missed' ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setFilter('repeated_missed')}
              style={{ fontSize: 11.5 }}
            >
              Repeated Missed Meds ({summary.repeated_missed_meds_count})
            </button>
            <button
              className={`btn btn-sm ${filter === 'worsening' ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setFilter('worsening')}
              style={{ fontSize: 11.5 }}
            >
              Worsening Symptoms ({summary.worsening_symptoms_count})
            </button>
          </div>

          <div style={{ position: 'relative', width: 240 }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              className="form-input"
              style={{ paddingLeft: 32, fontSize: 12 }}
              placeholder="Search patient, code, doctor..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Patients Table */}
        {filteredPatients.length === 0 ? (
          <div style={{ padding: '36px 20px', textAlign: 'center', color: 'var(--text-dim)' }}>
            No patients match the current filter criteria.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '10px 12px' }}>Patient Details</th>
                  <th style={{ padding: '10px 12px' }}>Assigned Specialist</th>
                  <th style={{ padding: '10px 12px' }}>Risk Level & Score</th>
                  <th style={{ padding: '10px 12px' }}>Trajectory Trend</th>
                  <th style={{ padding: '10px 12px' }}>Key Contributing Factors</th>
                  <th style={{ padding: '10px 12px' }}>Last Telemetry</th>
                  <th style={{ padding: '10px 12px' }}>Recommended Clinical Action</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map((p) => {
                  const rLvl = (p.risk_level || 'green').toLowerCase();
                  const badgeClass = rLvl === 'red' ? 'badge-red' : (rLvl === 'yellow' ? 'badge-amber' : (rLvl === 'green' ? 'badge-teal' : 'badge-neutral'));
                  const trendColor = p.trend_indicator === 'deteriorating' ? '#ef4444' : (p.trend_indicator === 'improving' ? '#22c55e' : 'var(--text-dim)');
                  const isExpanded = expandedPatientId === p.patient_id;

                  return (
                    <Fragment key={p.patient_id}>
                      <tr
                        style={{
                          borderBottom: isExpanded ? 'none' : '1px solid var(--border)',
                          background: rLvl === 'red' ? 'rgba(239, 68, 68, 0.04)' : isExpanded ? 'rgba(0, 212, 184, 0.03)' : 'transparent',
                          transition: 'background 0.2s ease'
                        }}
                      >
                        {/* Patient Details & Condition */}
                        <td style={{ padding: '12px' }}>
                          <div style={{ fontWeight: 700, color: 'var(--text)' }}>{p.name}</div>
                          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                            <code>{p.patient_code}</code> · {p.age ? `${p.age}y` : ''} {p.gender}
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
                            <span className="badge badge-teal" style={{ fontSize: 10, fontWeight: 700 }}>
                              {p.active_condition || 'Convalescence'}
                            </span>
                            {p.medical_history?.conditions?.length > 0 && (
                              <span className="badge badge-amber" style={{ fontSize: 9.5 }}>
                                {p.medical_history.conditions.length} Comorbidities
                              </span>
                            )}
                            {(p.custom_message_analysis || p.doctor_nurse_summary?.custom_notes_sentiment_flags?.length > 0) && (
                              <span className="badge badge-red" style={{ fontSize: 9.5, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                                <MessageSquare size={10} /> Note Signal
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Assigned Specialist */}
                        <td style={{ padding: '12px' }}>
                          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>{p.assigned_doctor}</div>
                        </td>

                        {/* Risk Level & Score */}
                        <td style={{ padding: '12px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span className={`badge ${badgeClass}`} style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase' }}>
                              {p.risk_level}
                            </span>
                            <span style={{ fontWeight: 800, fontSize: 13, color: 'var(--text)' }}>
                              {p.risk_score}
                            </span>
                          </div>
                          {p.is_override && (
                            <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
                              <ShieldAlert size={11} /> Overridden by MD
                            </div>
                          )}
                        </td>

                        {/* Trajectory Trend */}
                        <td style={{ padding: '12px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: trendColor, fontWeight: 700, textTransform: 'capitalize' }}>
                            {p.trend_indicator === 'deteriorating' && <TrendingDown size={14} />}
                            {p.trend_indicator === 'improving' && <TrendingUp size={14} />}
                            <span>{p.trend_indicator}</span>
                          </div>
                        </td>

                        {/* Key Contributing Factors */}
                        <td style={{ padding: '12px', maxWidth: 280 }}>
                          <ul style={{ margin: 0, paddingLeft: 16, fontSize: 11, color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {p.key_contributing_factors?.slice(0, 2).map((factor, idx) => (
                              <li key={idx}>{factor}</li>
                            ))}
                            {p.key_contributing_factors?.length > 2 && (
                              <li style={{ color: 'var(--p)', listStyle: 'none' }}>
                                +{p.key_contributing_factors.length - 2} more factors
                              </li>
                            )}
                          </ul>
                        </td>

                        {/* Last Telemetry */}
                        <td style={{ padding: '12px', fontSize: 11.5, color: 'var(--text-dim)' }}>
                          {p.last_checkin_date || 'No check-in'}
                        </td>

                        {/* Recommended Clinical Action */}
                        <td style={{ padding: '12px', maxWidth: 240 }}>
                          <div style={{ fontSize: 11.5, color: 'var(--text)', lineHeight: 1.35 }}>
                            {p.recommended_action}
                          </div>
                        </td>

                        {/* Actions */}
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, flexWrap: 'wrap' }}>
                            <button
                              className={`btn btn-sm ${isExpanded ? 'btn-primary' : 'btn-ghost'}`}
                              onClick={() => setExpandedPatientId(isExpanded ? null : p.patient_id)}
                              title="Toggle AI Clinical Assessment & Notes Deep-Dive"
                              style={{ padding: '3px 8px', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}
                            >
                              <Sparkles size={12} color={isExpanded ? '#fff' : 'var(--p)'} />
                              <span>AI Deep-Dive</span>
                              {isExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                            </button>
                            <button
                              className="btn btn-ghost btn-sm"
                              onClick={() => navigate && navigate(`#/staff/patients/${p.patient_id}/overview`)}
                              title="Open Patient Clinical Overview"
                              style={{ padding: '4px 6px' }}
                            >
                              <Eye size={13} />
                            </button>
                            <button
                              className="btn btn-ghost btn-sm"
                              onClick={() => handleReevaluate(p.patient_id)}
                              title="Force Re-evaluation"
                              style={{ padding: '4px 6px' }}
                            >
                              <RefreshCw size={13} />
                            </button>
                            <button
                              className="btn btn-ghost btn-sm"
                              onClick={() => {
                                setOverrideModal(p);
                                setOverrideLevel(p.risk_level);
                                setOverrideReason('');
                              }}
                              title="Clinician Risk Override"
                              style={{ padding: '4px 6px' }}
                            >
                              <Edit3 size={13} />
                            </button>
                          </div>
                        </td>
                      </tr>

                      {/* Expanded AI Clinical Deep-Dive Drawer */}
                      {isExpanded && (
                        <tr style={{ background: 'rgba(0, 212, 184, 0.02)' }}>
                          <td colSpan={8} style={{ padding: '16px 20px', borderBottom: '2px solid rgba(0, 212, 184, 0.4)' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                              {/* Header Banner */}
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10, borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                  <Sparkles size={16} color="var(--p)" />
                                  <span style={{ fontSize: 13.5, fontWeight: 800, color: 'var(--text)' }}>
                                    AI Multimodal Clinical Risk Assessment & Telemetry Analysis
                                  </span>
                                  <span className="badge badge-teal" style={{ fontSize: 10 }}>
                                    {p.active_condition} · {p.condition_stage || 'active'}
                                  </span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    Engine: <strong style={{ color: 'var(--p)' }}>{p.model_used || 'Gemini 3.5 Flash-Lite Clinical Risk Engine'}</strong>
                                  </span>
                                  {p.evaluated_at && (
                                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                      Evaluated: {new Date(p.evaluated_at).toLocaleString()}
                                    </span>
                                  )}
                                </div>
                              </div>

                              {/* 3-Column Diagnostic Breakdown */}
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
                                {/* Card 1: Medical History & Comorbidities */}
                                <div style={{ background: 'var(--surface2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <ClipboardList size={13} color="var(--p)" /> Medical History & Comorbidities
                                  </div>
                                  <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.45, marginBottom: 8 }}>
                                    {p.medical_history_impact || 'No adverse chronic disease interactions identified.'}
                                  </div>
                                  {p.medical_history?.conditions?.length > 0 && (
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                                      {p.medical_history.conditions.map((c, i) => (
                                        <span key={i} className="badge badge-amber" style={{ fontSize: 10 }}>{c}</span>
                                      ))}
                                    </div>
                                  )}
                                  {p.medical_history?.notes && (
                                    <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6, fontStyle: 'italic' }}>
                                      Notes: {p.medical_history.notes}
                                    </div>
                                  )}
                                </div>

                                {/* Card 2: Medication Adherence & Regimen */}
                                <div style={{ background: 'var(--surface2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <Pill size={13} color="var(--p)" /> Medication Adherence & Regimen
                                  </div>
                                  <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.45, marginBottom: 6 }}>
                                    {p.doctor_nurse_summary?.medication_impact || 'Prescription telemetry adheres to scheduled regimen.'}
                                  </div>
                                  {p.has_repeated_missed_meds ? (
                                    <div className="badge badge-red" style={{ fontSize: 10.5, marginTop: 4 }}>
                                      ⚠️ Repeated missed medication doses detected
                                    </div>
                                  ) : (
                                    <div className="badge badge-teal" style={{ fontSize: 10.5, marginTop: 4 }}>
                                      ✓ No critical missed medication streaks
                                    </div>
                                  )}
                                </div>

                                {/* Card 3: Telemetry & Patient Custom Note NLP */}
                                <div style={{ background: 'var(--surface2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <MessageSquare size={13} color="var(--p)" /> Symptom Trajectory & Patient Notes
                                  </div>
                                  <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.45, marginBottom: 6 }}>
                                    {p.doctor_nurse_summary?.symptom_trajectory_analysis || 'Telemetry and pain progression monitored.'}
                                  </div>
                                  {(p.custom_message_analysis || p.doctor_nurse_summary?.custom_notes_sentiment_flags?.length > 0) && (
                                    <div style={{ marginTop: 8, padding: 8, borderRadius: 6, background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
                                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 4 }}>
                                        {(p.doctor_nurse_summary?.custom_notes_sentiment_flags || []).map((fl, idx) => (
                                          <span key={idx} className="badge badge-red" style={{ fontSize: 9.5 }}>{fl}</span>
                                        ))}
                                      </div>
                                      {p.custom_message_analysis && (
                                        <div style={{ fontSize: 11.5, color: 'var(--text)', fontStyle: 'italic' }}>
                                          "{p.custom_message_analysis}"
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Full Clinical Narrative & Interventions */}
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12 }}>
                                {/* Clinical Assessment */}
                                <div style={{ background: 'var(--surface2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
                                    Synthesized Clinical Assessment (Clinician & Nursing)
                                  </div>
                                  <p style={{ margin: 0, fontSize: 12.5, color: 'var(--text)', lineHeight: 1.5 }}>
                                    {p.doctor_nurse_summary?.clinical_assessment || p.recommended_action}
                                  </p>
                                </div>

                                {/* Recommended Interventions */}
                                <div style={{ background: 'var(--surface2)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--p)', textTransform: 'uppercase', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <CheckCircle2 size={13} /> Actionable Clinical Interventions
                                  </div>
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                    {(p.doctor_nurse_summary?.recommended_clinical_interventions || [p.recommended_action]).map((act, i) => (
                                      <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 6, fontSize: 12, color: 'var(--text)' }}>
                                        <span style={{ color: 'var(--p)', fontWeight: 800 }}>•</span>
                                        <span>{act}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Clinician Override Modal */}
      {overrideModal && (
        <div className="modal-backdrop" onClick={() => setOverrideModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 480 }}>
            <div className="modal-header">
              <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700 }}>
                Clinician Risk Override: {overrideModal.name}
              </h3>
              <button className="btn btn-ghost btn-sm" onClick={() => setOverrideModal(null)}>
                <X size={16} />
              </button>
            </div>
            <form onSubmit={handleOverrideSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 14 }}>
              <div>
                <label className="form-label" style={{ fontSize: 12 }}>Target Clinical Risk Level</label>
                <select
                  className="form-input"
                  value={overrideLevel}
                  onChange={(e) => setOverrideLevel(e.target.value)}
                >
                  <option value="green">GREEN (Low Concern - Satisfactory Progression)</option>
                  <option value="yellow">YELLOW (Moderate Concern - Inconsistent / Monitored)</option>
                  <option value="red">RED (High Concern - Escalated Clinical Hazard)</option>
                  <option value="insufficient_data">INSUFFICIENT DATA (Telemetry Pending)</option>
                </select>
              </div>

              <div>
                <label className="form-label" style={{ fontSize: 12 }}>
                  Documented Clinical Rationale (Mandatory for Medical Audit)
                </label>
                <textarea
                  className="form-input"
                  rows={3}
                  placeholder="e.g., Patient contacted via telehealth; verified accidental double-entry resolved, wound is clean without erythema..."
                  value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  required
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 6 }}>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setOverrideModal(null)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary btn-sm">
                  Apply Clinician Override
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
