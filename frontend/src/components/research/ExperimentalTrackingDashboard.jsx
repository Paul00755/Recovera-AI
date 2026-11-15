import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, Users, RefreshCw, Download, RotateCcw, AlertTriangle,
  CheckCircle2, ChevronRight, FileText, Cpu, Pill, HeartPulse,
  TrendingDown, TrendingUp, ShieldAlert, Sparkles, ExternalLink
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function ExperimentalTrackingDashboard({ session, navigate }) {
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);
  const [cohortData, setCohortData] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [statusMessage, setStatusMessage] = useState(null);
  const [error, setError] = useState(null);

  const token = session?.token || JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}')?.token;
  const headers = useMemo(() => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }), [token]);

  const loadCohort = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/api/research/experimental/cohort`, { headers });
      if (!res.ok) throw new Error('Failed to load experimental cohort data');
      const data = await res.json();
      setCohortData(data);
      if (data.patients && data.patients.length > 0 && !selectedPatient) {
        setSelectedPatient(data.patients[0]);
      }
    } catch (err) {
      console.error('Cohort Loading Error:', err);
      setError(err.message || 'Error communicating with experimental cohort API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCohort();
  }, []);

  const handleResetCohort = async () => {
    const confirmed = window.confirm(
      'Reset all 5 experimental patients (P-101 to P-105) to initial ground-truth baseline states?\n\nThis will re-initialize clinical documents, OCR extractions, NLP entities, medication schedules, dosing occurrences, symptom check-ins, and digital twin states.'
    );
    if (!confirmed) return;

    try {
      setResetting(true);
      setStatusMessage('Resetting 5-patient cohort to pristine ground-truth states...');
      const res = await fetch(`${API_BASE}/api/research/experimental/reset`, {
        method: 'POST',
        headers
      });
      if (!res.ok) throw new Error('Cohort reset failed');
      await loadCohort();
      setStatusMessage('Experimental cohort successfully reset to baseline ground-truth.');
      setTimeout(() => setStatusMessage(null), 4000);
    } catch (err) {
      alert(`Reset Error: ${err.message}`);
    } finally {
      setResetting(false);
    }
  };

  const handleExportCSV = () => {
    window.open(`${API_BASE}/api/research/export/csv?data_type=experimental_cohort&token=${token}`, '_blank');
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px 20px', gap: 16 }}>
        <RefreshCw size={32} className="spin" style={{ color: 'var(--p)' }} />
        <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>
          Loading 5-Patient Experimental Cohort Matrix...
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, paddingBottom: 40 }}>
      {/* Header Banner */}
      <div className="card" style={{ padding: 24, background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span className="badge badge-teal" style={{ fontSize: 11, fontWeight: 700 }}>
                CONTROLLED EXPERIMENTAL STUDY
              </span>
              <span className="badge badge-neutral" style={{ fontSize: 11 }}>
                Cohort Size: 5 Longitudinal Clinical Trajectories
              </span>
              <span className="badge badge-amber" style={{ fontSize: 11 }}>
                Cross-Dimensional Analysis (10 Clinical Pillars)
              </span>
            </div>
            <h1 style={{ margin: '0 0 6px', fontSize: 24, fontWeight: 800, color: 'var(--text)' }}>
              5-Patient Longitudinal Experimental Tracking Dashboard
            </h1>
            <p style={{ margin: 0, fontSize: 13.5, color: 'var(--text-dim)', maxWidth: 840, lineHeight: 1.5 }}>
              Comprehensive longitudinal cohort evaluating genuine patient documents, OCR extraction accuracy, NLP medical entity normalization, adherence telemetry, symptom progressions, Digital Twin trajectory deviations, and multi-model machine learning predictions.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <button className="btn btn-ghost btn-sm" onClick={handleExportCSV}>
              <Download size={14} /> Export Cohort (CSV)
            </button>
            <button
              className="btn btn-danger btn-sm"
              onClick={handleResetCohort}
              disabled={resetting}
            >
              <RotateCcw size={14} className={resetting ? 'spin' : ''} />
              {resetting ? 'Resetting...' : 'Reset to Ground-Truth'}
            </button>
          </div>
        </div>

        {statusMessage && (
          <div className="alert alert-success" style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderRadius: 8 }}>
            <CheckCircle2 size={16} />
            <span style={{ fontSize: 13, fontWeight: 600 }}>{statusMessage}</span>
          </div>
        )}

        {error && (
          <div className="alert alert-danger" style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderRadius: 8 }}>
            <AlertTriangle size={16} />
            <span style={{ fontSize: 13 }}>{error}</span>
          </div>
        )}
      </div>

      {/* 10-Dimension Comparison Matrix Table */}
      <div className="card" style={{ padding: 22, background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="section-head" style={{ marginBottom: 16 }}>
          <div>
            <span className="eyebrow">LONGITUDINAL COMPARISON MATRIX</span>
            <h2 style={{ margin: '2px 0 0', fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>
              10-Dimensional Clinical Ground-Truth Evaluation Matrix
            </h2>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '10px 12px' }}>1. Patient / Code</th>
                <th style={{ padding: '10px 12px' }}>2. Condition & Stage</th>
                <th style={{ padding: '10px 12px' }}>3. Doc & OCR Status</th>
                <th style={{ padding: '10px 12px' }}>4. NLP Normalized Meds</th>
                <th style={{ padding: '10px 12px' }}>5. Adherence Telemetry</th>
                <th style={{ padding: '10px 12px' }}>6. Pain & Vitals</th>
                <th style={{ padding: '10px 12px' }}>7. Digital Twin Score</th>
                <th style={{ padding: '10px 12px' }}>8. Triage Risk Level</th>
                <th style={{ padding: '10px 12px' }}>9. ML Model Predictions</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>10. Action</th>
              </tr>
            </thead>
            <tbody>
              {cohortData?.patients?.map((p) => {
                const isSelected = selectedPatient?.patient_id === p.patient_id;
                const riskLvl = (p.risk_classification || 'GREEN').toLowerCase();
                const riskBadge = riskLvl === 'red' ? 'badge-red' : (riskLvl === 'yellow' ? 'badge-amber' : 'badge-teal');
                const dtBadge = p.digital_twin_deviation_tier === 'severe_deviation' ? 'badge-red' : (p.digital_twin_deviation_tier === 'mild_deviation' ? 'badge-amber' : 'badge-teal');

                // Extract RF and XGBoost predictions
                const rfPred = p.ml_model_predictions?.find(m => m.model_id === 'random_forest');
                const xgbPred = p.ml_model_predictions?.find(m => m.model_id === 'xgboost');
                const lstmPred = p.ml_model_predictions?.find(m => m.model_id === 'lstm_trajectory');

                return (
                  <tr
                    key={p.patient_id}
                    onClick={() => setSelectedPatient(p)}
                    style={{
                      borderBottom: '1px solid var(--border)',
                      cursor: 'pointer',
                      background: isSelected ? 'rgba(20, 184, 166, 0.08)' : 'transparent',
                      transition: 'background 0.15s ease'
                    }}
                  >
                    {/* 1. Patient / Code */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ fontWeight: 700, color: 'var(--text)' }}>{p.name}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        <code>{p.patient_code}</code> · {p.gender}
                      </div>
                    </td>

                    {/* 2. Condition & Stage */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ fontWeight: 600, color: 'var(--text)' }}>{p.condition || 'General Post-Op'}</div>
                      <span className={`badge ${p.condition_stage === 'acute' ? 'badge-red' : 'badge-neutral'}`} style={{ fontSize: 10, marginTop: 2 }}>
                        {p.condition_stage?.toUpperCase() || 'ROUTINE'}
                      </span>
                    </td>

                    {/* 3. Doc & OCR Status */}
                    <td style={{ padding: '12px' }}>
                      <span className={`badge ${p.document_extraction_status === 'approved' ? 'badge-teal' : 'badge-amber'}`} style={{ fontSize: 10 }}>
                        {p.document_extraction_status?.toUpperCase() || 'INGESTED'}
                      </span>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 3 }}>
                        {p.ocr_text_preview ? `${p.ocr_text_preview.slice(0, 32)}...` : 'Image verified'}
                      </div>
                    </td>

                    {/* 4. NLP Normalized Meds */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ fontWeight: 700, color: 'var(--text)' }}>
                        {p.nlp_extracted_medications_count} Medications
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)', maxWidth: 160, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {p.nlp_medications?.map(m => m.name).join(', ') || 'None'}
                      </div>
                    </td>

                    {/* 5. Adherence Telemetry */}
                    <td style={{ padding: '12px' }}>
                      {p.adherence_pct !== null && p.adherence_pct !== undefined ? (
                        <>
                          <div style={{ fontWeight: 700, color: p.adherence_pct < 60 ? '#ef4444' : (p.adherence_pct < 85 ? '#f59e0b' : '#22c55e') }}>
                            {p.adherence_pct}%
                          </div>
                          <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                            {p.total_occurrences} doses tracked
                          </div>
                        </>
                      ) : (
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Pending meds</span>
                      )}
                    </td>

                    {/* 6. Pain & Vitals */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontWeight: 700, color: (p.latest_pain_level || 0) >= 6 ? '#ef4444' : '#22c55e' }}>
                          Pain: {p.latest_pain_level != null ? `${p.latest_pain_level}/10` : 'N/A'}
                        </span>
                        {p.has_active_fever && (
                          <span className="badge badge-red" style={{ fontSize: 9.5, padding: '1px 4px' }}>Fever</span>
                        )}
                      </div>
                    </td>

                    {/* 7. Digital Twin Score */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontWeight: 800, fontSize: 13, color: 'var(--text)' }}>
                          {p.digital_twin_recovery_score}
                        </span>
                        <span className={`badge ${dtBadge}`} style={{ fontSize: 10 }}>
                          {p.digital_twin_deviation_tier?.replace(/_/g, ' ') || 'normal'}
                        </span>
                      </div>
                      <div style={{ fontSize: 10.5, color: 'var(--text-dim)' }}>
                        Dev: {p.composite_deviation}%
                      </div>
                    </td>

                    {/* 8. Triage Risk Level */}
                    <td style={{ padding: '12px' }}>
                      <span className={`badge ${riskBadge}`} style={{ fontSize: 10.5, fontWeight: 700 }}>
                        {p.risk_classification}
                      </span>
                    </td>

                    {/* 9. ML Model Predictions */}
                    <td style={{ padding: '12px' }}>
                      <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <div>RF: <strong style={{ color: rfPred?.predicted_risk_level === 'RED' ? '#ef4444' : '#22c55e' }}>{rfPred?.predicted_risk_level}</strong> ({rfPred?.probability}%)</div>
                        <div>XGB: <strong style={{ color: xgbPred?.predicted_risk_level === 'RED' ? '#ef4444' : '#22c55e' }}>{xgbPred?.predicted_risk_level}</strong> ({xgbPred?.probability}%)</div>
                        {lstmPred && <div style={{ color: 'var(--text-dim)' }}>LSTM: {lstmPred.predicted_value} pts</div>}
                      </div>
                    </td>

                    {/* 10. Action */}
                    <td style={{ padding: '12px', textAlign: 'center' }}>
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (navigate) navigate(`#/staff/patients/${p.patient_id}/overview`);
                        }}
                        title="View live patient dashboard"
                        style={{ padding: '4px 8px' }}
                      >
                        <ExternalLink size={13} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Patient Deep-Dive Detail View */}
      {selectedPatient && (
        <div className="card" style={{ padding: 22, background: 'var(--surface)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div>
              <span className="eyebrow">DETAILED CASE FILE INSPECTION</span>
              <h2 style={{ margin: '2px 0 0', fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>
                {selectedPatient.name} ({selectedPatient.patient_code}) — Clinical Trajectory Analysis
              </h2>
            </div>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => navigate && navigate(`#/staff/patients/${selectedPatient.patient_id}/overview`)}
            >
              Open Active Patient Workspace <ChevronRight size={14} />
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {/* Card 1: NLP Medical Entities Extracted */}
            <div style={{ padding: 16, borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Pill size={14} style={{ color: 'var(--p)' }} />
                <span>Verified Medications (Extracted via Hybrid NLP)</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {selectedPatient.nlp_medications?.length > 0 ? (
                  selectedPatient.nlp_medications.map((m, idx) => (
                    <div key={idx} style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--surface)', fontSize: 12 }}>
                      <div style={{ fontWeight: 700, color: 'var(--text)' }}>
                        {m.name} {m.strength}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        Dose: {m.dose} · Period: <span className="badge badge-teal" style={{ padding: '0 4px', fontSize: 10 }}>{m.period}</span> · {m.food_relation} · {m.duration_days} days
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No prescription medications ingested yet.</div>
                )}
              </div>
            </div>

            {/* Card 2: Risk Reasons & Explainability */}
            <div style={{ padding: 16, borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                <ShieldAlert size={14} style={{ color: 'var(--amber)' }} />
                <span>Clinical Risk Explainability Factors</span>
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: 6 }}>
                {selectedPatient.risk_reasons?.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
              {selectedPatient.key_drivers?.length > 0 && (
                <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
                  <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>
                    Machine Learning Key Input Drivers:
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11.5, color: 'var(--teal)' }}>
                    {selectedPatient.key_drivers.map((kd, idx) => (
                      <li key={idx}>{kd}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Card 3: Multi-Model Machine Learning Comparison */}
            <div style={{ padding: 16, borderRadius: 8, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Cpu size={14} style={{ color: 'var(--teal)' }} />
                <span>Live ML Predictions on Real Telemetry</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {selectedPatient.ml_model_predictions?.map((pred) => (
                  <div key={pred.model_id} style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--surface)', fontSize: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 700, color: 'var(--text)' }}>{pred.model_name}</span>
                      {pred.predicted_risk_level ? (
                        <span className={`badge ${pred.predicted_risk_level === 'RED' ? 'badge-red' : (pred.predicted_risk_level === 'YELLOW' ? 'badge-amber' : 'badge-teal')}`} style={{ fontSize: 10 }}>
                          {pred.predicted_risk_level} ({pred.probability}%)
                        </span>
                      ) : (
                        <span className="badge badge-neutral" style={{ fontSize: 10 }}>
                          {pred.predicted_value} pts
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 10.5, color: 'var(--text-dim)', marginTop: 2 }}>
                      Architecture: {pred.architecture} · Version: {pred.model_version}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
