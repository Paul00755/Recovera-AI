import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, ShieldCheck, RefreshCw, Layers, CheckCircle2,
  Clock, AlertTriangle, ArrowRight, User, Stethoscope, ChevronRight, FileText
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function DigitalTwinBenchmarkStudio({ session }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const token = session?.token || JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}')?.token;
  const headers = useMemo(() => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }), [token]);

  const fetchSummary = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/api/digital-twin/hospital-cohort-summary`, { headers });
      if (!res.ok) throw new Error('Failed to load clinical cohort telemetry');
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error loading clinical telemetry data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  const patients = data?.patients || [];
  const tiers = data?.tier_distribution || { normal: 0, mild_deviation: 0, moderate_deviation: 0, critical_disruption: 0 };

  const getTierBadge = (tier) => {
    switch (tier) {
      case 'critical_disruption':
        return { label: 'Critical Disruption', cls: 'badge-red' };
      case 'moderate_deviation':
        return { label: 'Moderate Deviation', cls: 'badge-amber' };
      case 'mild_deviation':
        return { label: 'Mild Deviation', cls: 'badge-teal' };
      default:
        return { label: 'Optimal Progression', cls: 'badge-green' };
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
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
              <Layers size={24} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span className="badge badge-teal" style={{ fontSize: 11, fontWeight: 700 }}>
                  ● CLINICAL OPERATIONS
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  14-Day Post-Discharge Telemetry
                </span>
              </div>
              <h1 style={{ margin: '4px 0 2px', fontSize: 20, fontWeight: 700, color: 'var(--text)' }}>
                Hospital Clinical Recovery Digital Twin Studio
              </h1>
              <p style={{ margin: 0, fontSize: 13, color: 'var(--text-dim)', maxWidth: 720 }}>
                Real-time recovery corridor tracking, multi-vector deviation scoring, and clinical intervention telemetry across all active post-discharge patients.
              </p>
            </div>
          </div>

          <button
            className="btn btn-ghost btn-sm"
            onClick={fetchSummary}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Refreshing...' : 'Refresh Telemetry'}
          </button>
        </div>

        {/* Real Hospital Telemetry Metric Strip */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginTop: 20 }}>
          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Active Monitored Patients
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
                {data?.total_patients ?? 0}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>patients</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Hospital inpatient & post-op registry
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Cohort Recovery Index (Rt)
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: 'var(--p)', fontVariantNumeric: 'tabular-nums' }}>
                {data?.average_recovery_score ?? 0}%
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>caseload avg</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Mathematical recovery score Rt = 100 × (1 - Dt)
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Medication Adherence Rate (At)
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: 'var(--success)', fontVariantNumeric: 'tabular-nums' }}>
                {data?.average_adherence ?? 0}%
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>confirmed logs</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Active prescription daily dose compliance
            </div>
          </div>

          <div className="stat-card" style={{ padding: '14px 18px', background: 'var(--surface2)', borderRadius: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Pending Deviation Flags
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 4 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: (data?.unreviewed_alerts || 0) > 0 ? 'var(--danger)' : 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
                {data?.unreviewed_alerts ?? 0}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>alerts</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              {(data?.unreviewed_alerts || 0) > 0 ? 'Requires clinician review' : 'All corridors stable'}
            </div>
          </div>
        </div>

        {/* Tier Distribution Summary */}
        <div style={{
          marginTop: 16,
          padding: '10px 14px',
          background: 'var(--surface2)',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 10,
          fontSize: 12
        }}>
          <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>
            Cohort Trajectory Tiers:
          </span>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--success)' }}></span>
              Optimal: <strong style={{ color: 'var(--text)' }}>{tiers.normal}</strong>
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#38bdf8' }}></span>
              Mild Deviation: <strong style={{ color: 'var(--text)' }}>{tiers.mild_deviation}</strong>
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--warning, #f59e0b)' }}></span>
              Moderate: <strong style={{ color: 'var(--text)' }}>{tiers.moderate_deviation}</strong>
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--danger)' }}></span>
              Critical Disruption: <strong style={{ color: 'var(--text)' }}>{tiers.critical_disruption}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Caseload Recovery Trajectory Matrix */}
      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>
              Hospital Caseload Recovery Trajectory Matrix
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: 12.5, color: 'var(--text-muted)' }}>
              Active patient trajectories evaluated against post-op clinical corridors and daily check-in logs.
            </p>
          </div>
          <span className="badge badge-neutral" style={{ fontSize: 11 }}>
            {patients.length} Monitored Patients
          </span>
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            {error}
          </div>
        )}

        {patients.length === 0 && !loading ? (
          <div style={{ padding: 36, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            No active post-discharge patients currently registered in the clinical workspace.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="table" style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left', color: 'var(--text-muted)', fontSize: 11.5 }}>
                  <th style={{ padding: '10px 12px' }}>PATIENT</th>
                  <th style={{ padding: '10px 12px' }}>ATTENDING CLINICIAN</th>
                  <th style={{ padding: '10px 12px' }}>RECOVERY TIMELINE</th>
                  <th style={{ padding: '10px 12px' }}>SCORE (Rt)</th>
                  <th style={{ padding: '10px 12px' }}>DEVIATION (Dt)</th>
                  <th style={{ padding: '10px 12px' }}>TELEMETRY VECTORS</th>
                  <th style={{ padding: '10px 12px' }}>CLINICAL STATUS</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right' }}>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p) => {
                  const tb = getTierBadge(p.deviation_tier);
                  return (
                    <tr
                      key={p.id}
                      style={{
                        borderBottom: '1px solid var(--border)',
                        transition: 'background 0.15s ease'
                      }}
                    >
                      <td style={{ padding: '12px 12px' }}>
                        <div style={{ fontWeight: 600, color: 'var(--text)' }}>
                          {p.name}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {p.patient_code} • {p.age ? `${p.age}y` : ''} {p.gender ? `(${p.gender})` : ''}
                        </div>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <div style={{ color: 'var(--text)', fontSize: 12.5 }}>
                          {p.doctor_name}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {p.specialization}
                        </div>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <Clock size={13} style={{ color: 'var(--text-muted)' }} />
                          <span style={{ fontWeight: 600, color: 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
                            Day {p.elapsed_day} of 14
                          </span>
                        </div>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <span style={{ fontWeight: 700, color: p.recovery_score >= 80 ? 'var(--p)' : 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
                          {p.recovery_score}%
                        </span>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <span style={{
                          fontWeight: 600,
                          color: p.composite_deviation > 25 ? 'var(--danger)' : p.composite_deviation > 15 ? 'var(--warning, #f59e0b)' : 'var(--text-dim)',
                          fontVariantNumeric: 'tabular-nums'
                        }}>
                          {p.composite_deviation}%
                        </span>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-dim)' }}>
                          <span title="Symptom divergence score">St: <strong style={{ color: 'var(--text)' }}>{p.symptom_score}%</strong></span>
                          <span>•</span>
                          <span title="Medication adherence gap">At: <strong style={{ color: 'var(--text)' }}>{p.adherence_score}%</strong></span>
                          <span>•</span>
                          <span title="Telemetry missingness">Qt: <strong style={{ color: 'var(--text)' }}>{p.missingness_score}%</strong></span>
                        </div>
                      </td>

                      <td style={{ padding: '12px 12px' }}>
                        <span className={`badge ${tb.cls}`} style={{ fontSize: 10 }}>
                          {tb.label}
                        </span>
                      </td>

                      <td style={{ padding: '12px 12px', textAlign: 'right' }}>
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs"
                          onClick={() => {
                            window.location.hash = `#/staff/patients/${p.id}/digital-twin`;
                          }}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                        >
                          <span>Open Twin</span>
                          <ChevronRight size={13} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Clinical Governance & Mathematical Formulation Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <Activity size={18} style={{ color: 'var(--p)' }} />
            <h4 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
              Deterministic State Formulation
            </h4>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: 'var(--text-dim)', lineHeight: 1.55 }}>
            Each patient's clinical digital twin continuously calculates a composite deviation score <code style={{ color: 'var(--p)', padding: '1px 4px', background: 'var(--surface2)', borderRadius: 4 }}>D_t = 0.35·S_t + 0.25·A_t + 0.20·T_t + 0.10·F_t + 0.10·Q_t</code>, mapping symptom divergence, medication adherence gaps, acute vitals, and missing logs against post-operative surgical corridors.
          </p>
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <ShieldCheck size={18} style={{ color: 'var(--success)' }} />
            <h4 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
              Supervised Clinical Governance
            </h4>
          </div>
          <p style={{ margin: 0, fontSize: 12.5, color: 'var(--text-dim)', lineHeight: 1.55 }}>
            Digital twin state monitoring operates strictly as an <em>advisory decision-support system</em>. Automated deviation flags notify attending physicians to review clinical progress, adjust care instructions, or schedule emergency consults without autonomous diagnosis.
          </p>
        </div>
      </div>
    </div>
  );
}
