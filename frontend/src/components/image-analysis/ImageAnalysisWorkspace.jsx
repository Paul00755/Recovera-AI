import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Camera, UploadCloud, RefreshCw, CheckCircle2, AlertTriangle, AlertCircle,
  FileText, Download, ChevronRight, Eye, ShieldAlert, Sparkles, Activity,
  Sliders, Layers, Check, X, ArrowLeft, ZoomIn, ZoomOut, Maximize2,
  Calendar, User, Stethoscope, Clock, ShieldCheck, BarChart3, HelpCircle
} from 'lucide-react';
import { getApiBase } from '../../apiConfig';

const API_BASE = getApiBase();

async function clientApi(path, options = {}) {
  const base = getApiBase();
  const getStoredToken = () => {
    try {
      const stored = JSON.parse(sessionStorage.getItem('recovera_session') || localStorage.getItem('recovera_session') || '{}');
      return stored?.token || stored?.access_token || '';
    } catch {
      return '';
    }
  };
  const token = getStoredToken();

  const headers = { ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${base}${path}`, { ...options, headers });
  if (!res.ok) {
    let msg = 'Request failed';
    try {
      const j = await res.json();
      msg = j.detail || msg;
    } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

export function ImageAnalysisWorkspace({ session, navigate, patients: propPatients = [] }) {
  const isDoctor = session?.user?.role === 'doctor';
  const isNurse = session?.user?.role === 'nurse';
  const isAdmin = session?.user?.role === 'admin';

  // State - initialized immediately with assigned patients from StaffShell if available
  const [patients, setPatients] = useState(() => (Array.isArray(propPatients) && propPatients.length > 0 ? propPatients : []));
  const [loadingPatients, setLoadingPatients] = useState(() => !(Array.isArray(propPatients) && propPatients.length > 0));
  const [selectedPatientId, setSelectedPatientId] = useState(() => (Array.isArray(propPatients) && propPatients.length > 0 ? propPatients[0].id : ''));
  const [searchQuery, setSearchQuery] = useState('');

  // Modality & Upload state
  const [analysisType, setAnalysisType] = useState('surgical_wound'); // surgical_wound | bone_fracture_xray
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [anatomicalSite, setAnatomicalSite] = useState('');
  const [clinicalView, setClinicalView] = useState('');
  const [isBaseline, setIsBaseline] = useState(false);
  const [baselineImageId, setBaselineImageId] = useState('');

  // Execution state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sessionData, setSessionData] = useState(null); // active draft session
  const [activeAnalysis, setActiveAnalysis] = useState(null); // current ImageAnalysis payload
  const [patientHistory, setPatientHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // Active View Tab on Canvas
  // 'original' | 'gradcam_overlay' | 'gradcam_heatmap' | 'segmentation' | 'longitudinal'
  const [activeCanvasTab, setActiveCanvasTab] = useState('gradcam_overlay');
  const [zoomLevel, setZoomLevel] = useState(1.0);

  // Doctor Review state
  const [reviewStatus, setReviewStatus] = useState('approved');
  const [clinicianNotes, setClinicianNotes] = useState('');
  const [clinicianAgreed, setClinicianAgreed] = useState(true);
  const [doctorFindings, setDoctorFindings] = useState('');
  const [isPatientVisible, setIsPatientVisible] = useState(false);
  const [submittingReview, setSubmittingReview] = useState(false);
  const [editReviewMode, setEditReviewMode] = useState(false);

  // Benchmarks Modal
  const [benchmarksOpen, setBenchmarksOpen] = useState(false);
  const [benchmarkRuns, setBenchmarkRuns] = useState([]);
  const [loadingBenchmarks, setLoadingBenchmarks] = useState(false);

  // Pending reviews count
  const [pendingReviews, setPendingReviews] = useState([]);

  // Notifications / Feedback
  const [feedback, setFeedback] = useState(null);
  const fileInputRef = useRef(null);

  // 1. Load Patients & Pending Reviews
  const loadPatients = async () => {
    try {
      setLoadingPatients(true);
      let data = null;
      try {
        data = await clientApi('/api/image-analysis/patients');
      } catch (err) {
        console.warn('Image analysis patients endpoint error, attempting /api/patients fallback:', err);
      }
      if (!data || !Array.isArray(data) || data.length === 0) {
        try {
          data = await clientApi('/api/patients');
        } catch (err) {
          console.warn('Fallback /api/patients error:', err);
        }
      }
      if ((!data || data.length === 0) && Array.isArray(propPatients) && propPatients.length > 0) {
        data = propPatients;
      }
      if (Array.isArray(data) && data.length > 0) {
        setPatients(data);
        setSelectedPatientId(prev => prev || data[0].id);
      }
    } catch (e) {
      console.error('Failed to load patients for image analysis:', e);
      if (Array.isArray(propPatients) && propPatients.length > 0) {
        setPatients(propPatients);
        setSelectedPatientId(prev => prev || propPatients[0].id);
      }
    } finally {
      setLoadingPatients(false);
    }
  };

  useEffect(() => {
    if (Array.isArray(propPatients) && propPatients.length > 0) {
      setPatients(prev => {
        if (!prev || prev.length === 0) {
          setSelectedPatientId(pId => pId || propPatients[0].id);
          return propPatients;
        }
        return prev;
      });
    }
  }, [propPatients]);

  const loadPendingReviews = async () => {
    try {
      const data = await clientApi('/api/image-analysis/pending-reviews');
      setPendingReviews(data || []);
    } catch (e) {
      // non-blocking
    }
  };

  useEffect(() => {
    loadPatients();
    loadPendingReviews();
  }, []);

  // 2. Load Selected Patient History
  const loadHistory = async (pid) => {
    if (!pid) return;
    try {
      setLoadingHistory(true);
      const hist = await clientApi(`/api/image-analysis/patients/${pid}/history`);
      setPatientHistory(hist || []);
      // If there are existing saved analyses and no active draft, preview latest
      if (hist && hist.length > 0 && !sessionData) {
        setActiveAnalysis(hist[0]);
        setReviewStatus(hist[0].review_status || 'approved');
        setClinicianNotes(hist[0].clinician_notes || '');
        setClinicianAgreed(hist[0].clinician_agreed_with_ai ?? true);
        setDoctorFindings(hist[0].doctor_confirmed_findings || '');
        setIsPatientVisible(hist[0].is_patient_visible || false);
      }
    } catch (e) {
      console.error('Failed to load patient history:', e);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (selectedPatientId) {
      loadHistory(selectedPatientId);
      // Auto-populate anatomical defaults based on patient condition
      const pat = patients.find(p => p.id === selectedPatientId);
      if (pat) {
        const cond = (pat.active_condition || '').toLowerCase();
        if (cond.includes('fracture') || cond.includes('bone') || cond.includes('tibia') || cond.includes('radius')) {
          setAnalysisType('bone_fracture_xray');
          setAnatomicalSite(cond.includes('tibia') ? 'Right Tibia / Fibula' : (cond.includes('radius') ? 'Left Distal Radius' : 'Extremity Bone'));
          setClinicalView('AP & Lateral Radiograph');
        } else {
          setAnalysisType('surgical_wound');
          setAnatomicalSite(cond.includes('appendectomy') ? 'Lower Right Abdominal Incision' : (cond.includes('knee') ? 'Anterior Midline Knee' : 'Surgical Incision Site'));
          setClinicalView('Close-up Tangential');
        }
      }
    }
  }, [selectedPatientId, patients]);

  const selectedPatient = useMemo(() => {
    return patients.find(p => p.id === selectedPatientId);
  }, [patients, selectedPatientId]);

  // File selection & preview
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setFilePreview(url);
    setFeedback(null);
  };

  // 3. Initiate Deep Learning Analysis
  const handleInitiateAnalysis = async () => {
    if (!selectedPatientId || !selectedFile) {
      setFeedback({ type: 'error', message: 'Please select a patient and upload an image.' });
      return;
    }

    try {
      setIsAnalyzing(true);
      setFeedback(null);

      const formData = new FormData();
      formData.append('patient_id', selectedPatientId);
      formData.append('analysis_type', analysisType);
      formData.append('file', selectedFile);
      formData.append('anatomical_site', anatomicalSite || 'General Region');
      formData.append('clinical_view', clinicalView || 'Standard View');
      formData.append('is_baseline', isBaseline ? 'true' : 'false');
      if (baselineImageId) formData.append('baseline_image_id', baselineImageId);

      const res = await clientApi('/api/image-analysis/sessions', {
        method: 'POST',
        body: formData
      });

      setSessionData({ id: res.session_id, status: res.session_status });
      setActiveAnalysis(res.analysis);
      setActiveCanvasTab('gradcam_overlay');
      setFeedback({
        type: 'success',
        message: `Deep learning inference complete (${res.analysis.processing_time_ms}ms). Session in DRAFT mode.`
      });
    } catch (e) {
      setFeedback({ type: 'error', message: e.message || 'Analysis failed. Please check image criteria.' });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 4. Save Session (Commit to Patient Care & trigger Gemini)
  const handleSaveSession = async () => {
    if (!sessionData?.id) return;
    try {
      setIsAnalyzing(true);
      const res = await clientApi(`/api/image-analysis/sessions/${sessionData.id}/save`, {
        method: 'POST'
      });
      setSessionData(null); // No longer in draft mode
      setActiveAnalysis(res.analysis);
      setFeedback({
        type: 'success',
        message: 'Analysis saved and integrated into Patient Care Trajectory. Gemini reassessment generated.'
      });
      loadHistory(selectedPatientId);
      loadPendingReviews();
    } catch (e) {
      setFeedback({ type: 'error', message: e.message || 'Failed to save analysis session.' });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 5. Discard Draft Session
  const handleDiscardSession = async () => {
    if (!sessionData?.id) return;
    if (!window.confirm('Discard this draft analysis? No clinical records will be altered.')) return;
    try {
      await clientApi(`/api/image-analysis/sessions/${sessionData.id}/discard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Clinician discarded preliminary draft.' })
      });
      setSessionData(null);
      setSelectedFile(null);
      setFilePreview(null);
      setFeedback({ type: 'info', message: 'Draft analysis session discarded.' });
      loadHistory(selectedPatientId);
    } catch (e) {
      setFeedback({ type: 'error', message: e.message });
    }
  };

  // 6. Doctor Sign-Off & Review Submission
  const handleReviewSubmit = async () => {
    if (!activeAnalysis?.id) return;
    try {
      setSubmittingReview(true);
      const res = await clientApi(`/api/image-analysis/${activeAnalysis.id}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review_status: reviewStatus,
          clinician_notes: clinicianNotes,
          clinician_agreed_with_ai: clinicianAgreed,
          doctor_confirmed_findings: doctorFindings || activeAnalysis.findings_summary,
          is_patient_visible: isPatientVisible
        })
      });
      setActiveAnalysis(res.analysis);
      setEditReviewMode(false);
      setFeedback({
        type: 'success',
        message: `Clinical review signed off as ${reviewStatus.toUpperCase()}.`
      });
      loadHistory(selectedPatientId);
      loadPendingReviews();
    } catch (e) {
      setFeedback({ type: 'error', message: e.message || 'Failed to sign off review.' });
    } finally {
      setSubmittingReview(false);
    }
  };

  // 7. Load Research Benchmarks
  const handleOpenBenchmarks = async () => {
    setBenchmarksOpen(true);
    try {
      setLoadingBenchmarks(true);
      const runs = await clientApi('/api/image-analysis/benchmarks/runs');
      setBenchmarkRuns(runs || []);
    } catch (e) {
      console.error('Failed to load benchmarks:', e);
    } finally {
      setLoadingBenchmarks(false);
    }
  };

  // PDF Export
  const handleExportPdf = () => {
    if (!activeAnalysis?.id) return;
    const sessionStr = localStorage.getItem('recovera_session');
    let token = '';
    try { token = JSON.parse(sessionStr)?.token || ''; } catch (e) {}
    window.open(`${API_BASE}/api/image-analysis/${activeAnalysis.id}/export/pdf?token=${encodeURIComponent(token)}`, '_blank');
  };

  // CSV Export
  const handleExportCsv = () => {
    if (!activeAnalysis?.id) return;
    const sessionStr = localStorage.getItem('recovera_session');
    let token = '';
    try { token = JSON.parse(sessionStr)?.token || ''; } catch (e) {}
    window.open(`${API_BASE}/api/image-analysis/${activeAnalysis.id}/export/csv?token=${encodeURIComponent(token)}`, '_blank');
  };

  // Helper for rendering image artifact URLs
  const getRenderUrl = (relPath) => {
    if (!relPath) return null;
    if (relPath.startsWith('http://') || relPath.startsWith('https://') || relPath.startsWith('blob:') || relPath.startsWith('data:')) {
      return relPath;
    }
    const base = getApiBase();
    const cleanRel = relPath.startsWith('/') ? relPath : `/${relPath}`;
    return `${base}${cleanRel}`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 'calc(100vh - 120px)' }}>

      {/* Top Clinical Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--card-bg, #0c131d)',
        border: '1px solid var(--border-color, #1a2636)',
        borderRadius: 12,
        padding: '12px 18px',
        flexWrap: 'wrap',
        gap: 12
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 38,
            height: 38,
            borderRadius: 8,
            background: 'rgba(0, 212, 184, 0.12)',
            color: '#00d4b8',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Camera size={20} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0, color: 'var(--text-primary, #f1f5f9)' }}>
                AI Medical Image Analysis & Deep Learning
              </h2>
              <span style={{
                fontSize: 11,
                padding: '2px 8px',
                borderRadius: 12,
                fontWeight: 600,
                background: analysisType === 'surgical_wound' ? 'rgba(0, 212, 184, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                color: analysisType === 'surgical_wound' ? '#00d4b8' : '#38bdf8',
                border: `1px solid ${analysisType === 'surgical_wound' ? 'rgba(0, 212, 184, 0.3)' : 'rgba(56, 189, 248, 0.3)'}`
              }}>
                {analysisType === 'surgical_wound' ? 'Wound Monitoring (ResNet50 + U-Net++)' : 'Bone Radiograph (DenseNet121)'}
              </span>
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary, #94a3b8)', margin: '2px 0 0' }}>
              Multi-scale computer vision, Grad-CAM neural attribution, U-Net++ segmentation & Gemini recovery synthesis
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {pendingReviews.length > 0 && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(245, 158, 11, 0.12)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              color: '#f59e0b',
              padding: '6px 12px',
              borderRadius: 8,
              fontSize: 12,
              fontWeight: 600
            }}>
              <AlertCircle size={14} />
              <span>{pendingReviews.length} Pending Review{pendingReviews.length > 1 ? 's' : ''}</span>
            </div>
          )}

          <button
            onClick={handleOpenBenchmarks}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 14px',
              borderRadius: 8,
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-color, #223348)',
              color: 'var(--text-primary, #f1f5f9)',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <BarChart3 size={14} />
            <span>Research Benchmarks</span>
          </button>

          {activeAnalysis && (
            <>
              <button
                onClick={handleExportPdf}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '8px 14px',
                  borderRadius: 8,
                  background: 'rgba(0, 212, 184, 0.1)',
                  border: '1px solid rgba(0, 212, 184, 0.3)',
                  color: '#00d4b8',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                <FileText size={14} />
                <span>Export PDF Dossier</span>
              </button>
              <button
                onClick={handleExportCsv}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '8px 12px',
                  borderRadius: 8,
                  background: 'transparent',
                  border: '1px solid var(--border-color, #223348)',
                  color: 'var(--text-secondary, #94a3b8)',
                  fontSize: 12,
                  cursor: 'pointer'
                }}
                title="Download CSV Biomarkers"
              >
                <Download size={14} />
                <span>CSV</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Action / Error Banner */}
      {feedback && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          borderRadius: 8,
          fontSize: 13,
          background: feedback.type === 'error' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(0, 212, 184, 0.15)',
          color: feedback.type === 'error' ? '#ef4444' : '#00d4b8',
          border: `1px solid ${feedback.type === 'error' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(0, 212, 184, 0.3)'}`
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {feedback.type === 'error' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
            <span>{feedback.message}</span>
          </div>
          <button
            onClick={() => setFeedback(null)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Main 3-Panel Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '320px 1fr 340px',
        gap: 16,
        alignItems: 'start'
      }}>

        {/* ------------------------------------------------------------------ */}
        {/* LEFT PANEL: Patient Selector & Upload Workspace                    */}
        {/* ------------------------------------------------------------------ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Patient Selector Card */}
          <div style={{
            background: 'var(--card-bg, #0c131d)',
            border: '1px solid var(--border-color, #1a2636)',
            borderRadius: 12,
            padding: 14
          }}>
            <label style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 6 }}>
              Select Active Patient
            </label>
            <select
              value={selectedPatientId}
              onChange={(e) => {
                setSelectedPatientId(e.target.value);
                setSessionData(null);
                setSelectedFile(null);
                setFilePreview(null);
              }}
              style={{
                width: '100%',
                padding: '9px 12px',
                borderRadius: 8,
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-color, #223348)',
                color: 'var(--text-primary, #f1f5f9)',
                fontSize: 13,
                outline: 'none'
              }}
            >
              {patients.length === 0 ? (
                <option value="" disabled style={{ background: '#0c131d', color: '#94a3b8' }}>
                  {loadingPatients ? 'Loading assigned patients...' : 'No assigned patients available'}
                </option>
              ) : (
                patients.map(p => (
                  <option key={p.id} value={p.id} style={{ background: '#0c131d', color: '#fff' }}>
                    {p.name || 'Patient'} ({p.patient_code || 'REC-PT'}) — {p.active_condition ? p.active_condition.slice(0, 32) : 'General Care'}
                  </option>
                ))
              )}
            </select>

            {/* Selected Patient Dossier */}
            {selectedPatient && (
              <div style={{
                marginTop: 10,
                padding: 10,
                borderRadius: 8,
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--border-color, #1a2636)',
                fontSize: 12
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-secondary, #94a3b8)' }}>MRN:</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary, #f1f5f9)' }}>{selectedPatient.patient_code}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-secondary, #94a3b8)' }}>Active Condition:</span>
                  <span style={{ fontWeight: 600, color: '#00d4b8', textAlign: 'right', maxWidth: 170 }}>{selectedPatient.active_condition || 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-secondary, #94a3b8)' }}>Attending Physician:</span>
                  <span style={{ color: 'var(--text-primary, #f1f5f9)' }}>{selectedPatient.assigned_doctor_name || 'Dr. Assigned'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary, #94a3b8)' }}>Baseline Status:</span>
                  <span style={{
                    color: (selectedPatient.has_baseline_wound || selectedPatient.has_baseline_xray) ? '#10b981' : '#f59e0b',
                    fontWeight: 600
                  }}>
                    {(selectedPatient.has_baseline_wound || selectedPatient.has_baseline_xray) ? 'Established' : 'Pending Baseline'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Analysis Configuration & Upload Card */}
          <div style={{
            background: 'var(--card-bg, #0c131d)',
            border: '1px solid var(--border-color, #1a2636)',
            borderRadius: 12,
            padding: 14,
            display: 'flex',
            flexDirection: 'column',
            gap: 12
          }}>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-secondary, #94a3b8)', display: 'block', marginBottom: 6 }}>
                Image Modality
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <button
                  type="button"
                  onClick={() => setAnalysisType('surgical_wound')}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 8,
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: analysisType === 'surgical_wound' ? 'rgba(0, 212, 184, 0.15)' : 'rgba(0,0,0,0.2)',
                    color: analysisType === 'surgical_wound' ? '#00d4b8' : 'var(--text-secondary, #94a3b8)',
                    border: `1px solid ${analysisType === 'surgical_wound' ? '#00d4b8' : 'var(--border-color, #223348)'}`
                  }}
                >
                  Surgical Wound
                </button>
                <button
                  type="button"
                  onClick={() => setAnalysisType('bone_fracture_xray')}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 8,
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: analysisType === 'bone_fracture_xray' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(0,0,0,0.2)',
                    color: analysisType === 'bone_fracture_xray' ? '#38bdf8' : 'var(--text-secondary, #94a3b8)',
                    border: `1px solid ${analysisType === 'bone_fracture_xray' ? '#38bdf8' : 'var(--border-color, #223348)'}`
                  }}
                >
                  Bone Fracture (X-Ray)
                </button>
              </div>
            </div>

            {/* Baseline Checkbox */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-primary, #f1f5f9)' }}>
              <input
                type="checkbox"
                id="isBaselineChk"
                checked={isBaseline}
                onChange={(e) => setIsBaseline(e.target.checked)}
                style={{ cursor: 'pointer', accentColor: '#00d4b8' }}
              />
              <label htmlFor="isBaselineChk" style={{ cursor: 'pointer' }}>
                Set as Baseline Reference Image
              </label>
            </div>

            {/* Upload Dropzone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border-color, #223348)',
                borderRadius: 8,
                padding: '20px 14px',
                textAlign: 'center',
                cursor: 'pointer',
                background: selectedFile ? 'rgba(0, 212, 184, 0.04)' : 'rgba(0,0,0,0.2)',
                transition: 'all 0.2s'
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />

              {filePreview ? (
                <div>
                  <img
                    src={filePreview}
                    alt="Upload Preview"
                    style={{ maxHeight: 110, maxWidth: '100%', borderRadius: 6, objectFit: 'contain', marginBottom: 8 }}
                  />
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#00d4b8' }}>{selectedFile?.name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Click to replace file</div>
                </div>
              ) : (
                <div>
                  <UploadCloud size={28} style={{ color: 'var(--text-secondary, #94a3b8)', marginBottom: 6 }} />
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary, #f1f5f9)' }}>
                    Drop patient image or click to browse
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                    {analysisType === 'bone_fracture_xray' ? 'Authentic X-ray Radiograph (Grayscale)' : 'Clinical Photograph (JPEG/PNG)'}
                  </div>
                </div>
              )}
            </div>

            {/* Run Button */}
            <button
              onClick={handleInitiateAnalysis}
              disabled={isAnalyzing || !selectedFile}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '11px',
                borderRadius: 8,
                background: (!selectedFile || isAnalyzing) ? 'rgba(255,255,255,0.08)' : '#00d4b8',
                color: (!selectedFile || isAnalyzing) ? 'var(--text-secondary, #94a3b8)' : '#041017',
                fontWeight: 700,
                fontSize: 13,
                cursor: (!selectedFile || isAnalyzing) ? 'not-allowed' : 'pointer',
                border: 'none',
                transition: 'all 0.2s'
              }}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw size={16} className="spin-animate" />
                  <span>Executing Vision Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Initiate AI Vision Analysis</span>
                </>
              )}
            </button>
          </div>

          {/* Historical Saved Images Timeline Card */}
          <div style={{
            background: 'var(--card-bg, #0c131d)',
            border: '1px solid var(--border-color, #1a2636)',
            borderRadius: 12,
            padding: 14
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <label style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-secondary, #94a3b8)' }}>
                Patient Image Records ({patientHistory.length})
              </label>
              <button
                onClick={() => loadHistory(selectedPatientId)}
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}
                title="Refresh history"
              >
                <RefreshCw size={12} />
              </button>
            </div>

            {loadingHistory ? (
              <div style={{ padding: '20px 0', textAlign: 'center', fontSize: 12, color: 'var(--text-secondary, #94a3b8)' }}>
                Loading image history...
              </div>
            ) : patientHistory.length === 0 ? (
              <div style={{ padding: '16px 0', textAlign: 'center', fontSize: 12, color: 'var(--text-secondary, #94a3b8)' }}>
                No saved image analyses yet.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 220, overflowY: 'auto' }}>
                {patientHistory.map((item) => {
                  const isCurrent = activeAnalysis?.id === item.id;
                  const thumb = item.artifacts?.gradcam_overlay_url || item.image?.url;
                  return (
                    <div
                      key={item.id}
                      onClick={() => {
                        setActiveAnalysis(item);
                        setSessionData(null);
                        setReviewStatus(item.review_status || 'approved');
                        setClinicianNotes(item.clinician_notes || '');
                        setClinicianAgreed(item.clinician_agreed_with_ai ?? true);
                        setDoctorFindings(item.doctor_confirmed_findings || '');
                        setIsPatientVisible(item.is_patient_visible || false);
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                        padding: 8,
                        borderRadius: 8,
                        cursor: 'pointer',
                        background: isCurrent ? 'rgba(0, 212, 184, 0.12)' : 'rgba(255,255,255,0.02)',
                        border: `1px solid ${isCurrent ? '#00d4b8' : 'var(--border-color, #1a2636)'}`
                      }}
                    >
                      {thumb ? (
                        <img
                          src={getRenderUrl(thumb)}
                          alt="Thumbnail"
                          style={{ width: 44, height: 44, borderRadius: 6, objectFit: 'cover' }}
                        />
                      ) : (
                        <div style={{ width: 44, height: 44, borderRadius: 6, background: '#1a2636', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Camera size={16} color="#64748b" />
                        </div>
                      )}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary, #f1f5f9)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {item.primary_prediction}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', display: 'flex', gap: 6, marginTop: 2 }}>
                          <span>{item.image?.is_baseline ? 'Baseline' : 'Follow-up'}</span>
                          <span>&bull;</span>
                          <span>{(item.confidence_score * 100).toFixed(0)}% conf</span>
                        </div>
                      </div>
                      <span style={{
                        fontSize: 10,
                        fontWeight: 600,
                        padding: '2px 6px',
                        borderRadius: 10,
                        background: item.review_status === 'approved' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                        color: item.review_status === 'approved' ? '#10b981' : '#f59e0b'
                      }}>
                        {item.review_status === 'approved' ? 'Approved' : 'Pending'}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

        </div>

        {/* ------------------------------------------------------------------ */}
        {/* CENTER PANEL: Multimodal Vision & Explainability Canvas             */}
        {/* ------------------------------------------------------------------ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Canvas Wrapper */}
          <div style={{
            background: 'var(--card-bg, #0c131d)',
            border: '1px solid var(--border-color, #1a2636)',
            borderRadius: 12,
            overflow: 'hidden'
          }}>

            {/* Canvas Header & View Switcher Tabs */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              borderBottom: '1px solid var(--border-color, #1a2636)',
              background: 'rgba(0,0,0,0.2)',
              flexWrap: 'wrap',
              gap: 8
            }}>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <button
                  key="gradcam_overlay"
                  onClick={() => setActiveCanvasTab('gradcam_overlay')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '6px 12px',
                    borderRadius: 6,
                    fontSize: 12,
                    fontWeight: 700,
                    cursor: 'pointer',
                    background: activeCanvasTab === 'gradcam_overlay' ? '#00d4b8' : 'rgba(255,255,255,0.05)',
                    color: activeCanvasTab === 'gradcam_overlay' ? '#041017' : 'var(--text-secondary, #94a3b8)',
                    border: 'none',
                    transition: 'all 0.15s'
                  }}
                >
                  <Layers size={13} />
                  <span>Grad-CAM Overlay</span>
                </button>
                <button
                  key="original"
                  onClick={() => setActiveCanvasTab('original')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '6px 12px',
                    borderRadius: 6,
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: activeCanvasTab === 'original' ? '#00d4b8' : 'rgba(255,255,255,0.05)',
                    color: activeCanvasTab === 'original' ? '#041017' : 'var(--text-secondary, #94a3b8)',
                    border: 'none',
                    transition: 'all 0.15s'
                  }}
                >
                  <Eye size={13} />
                  <span>Original Clinical Image</span>
                </button>
              </div>

              {/* Zoom Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <button
                  onClick={() => setZoomLevel(prev => Math.max(0.6, prev - 0.2))}
                  style={{ padding: 5, borderRadius: 4, background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color, #223348)', color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}
                  title="Zoom Out"
                >
                  <ZoomOut size={13} />
                </button>
                <span style={{ fontSize: 11, minWidth: 36, textAlign: 'center', color: 'var(--text-secondary, #94a3b8)' }}>
                  {Math.round(zoomLevel * 100)}%
                </span>
                <button
                  onClick={() => setZoomLevel(prev => Math.min(2.5, prev + 0.2))}
                  style={{ padding: 5, borderRadius: 4, background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color, #223348)', color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}
                  title="Zoom In"
                >
                  <ZoomIn size={13} />
                </button>
                <button
                  onClick={() => setZoomLevel(1.0)}
                  style={{ padding: '4px 8px', borderRadius: 4, background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color, #223348)', color: 'var(--text-secondary, #94a3b8)', fontSize: 11, cursor: 'pointer' }}
                >
                  Fit
                </button>
              </div>
            </div>

            {/* Canvas Viewport */}
            <div style={{
              height: 440,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#04070c',
              position: 'relative',
              overflow: 'hidden'
            }}>
              {!activeAnalysis && !filePreview ? (
                <div style={{ textAlign: 'center', color: 'var(--text-secondary, #94a3b8)', padding: 20 }}>
                  <Camera size={44} style={{ opacity: 0.3, marginBottom: 10 }} />
                  <div style={{ fontSize: 14, fontWeight: 600 }}>No Medical Image Loaded</div>
                  <div style={{ fontSize: 12, opacity: 0.8, marginTop: 4 }}>
                    Select a patient from the left panel and upload an image, or click a historical record.
                  </div>
                </div>
              ) : (
                /* Single Image Mode with Zoom */
                <div style={{
                  transform: `scale(${zoomLevel})`,
                  transition: 'transform 0.15s ease',
                  maxHeight: '100%',
                  maxWidth: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {(() => {
                    let src = null;
                    if (activeCanvasTab === 'gradcam_overlay') {
                      src = getRenderUrl(activeAnalysis?.artifacts?.gradcam_overlay_url) || filePreview;
                    } else {
                      src = getRenderUrl(activeAnalysis?.image?.url) || filePreview;
                    }
                    return (
                      <img
                        src={src}
                        alt="Clinical Display"
                        style={{ maxHeight: 420, maxWidth: '100%', objectFit: 'contain', borderRadius: 6 }}
                      />
                    );
                  })()}
                </div>
              )}

              {/* Attribution Overlay Label Tag */}
              {activeAnalysis && (
                <div style={{
                  position: 'absolute',
                  bottom: 12,
                  left: 12,
                  background: 'rgba(0,0,0,0.85)',
                  backdropFilter: 'blur(6px)',
                  border: '1px solid rgba(0, 212, 184, 0.3)',
                  borderRadius: 8,
                  padding: '6px 12px',
                  fontSize: 11.5,
                  color: 'var(--text-secondary, #94a3b8)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#00d4b8', fontWeight: 700 }}>
                    <Layers size={13} />
                    <span>Grad-CAM</span>
                  </span>
                  <span>&bull;</span>
                  <span>Disease: <b style={{ color: '#fff' }}>{activeAnalysis.primary_prediction}</b></span>
                  <span>&bull;</span>
                  <span>Confidence: <b style={{ color: '#00d4b8' }}>{(activeAnalysis.confidence_score * 100).toFixed(1)}%</b></span>
                  <span>&bull;</span>
                  <span>Model: <b style={{ color: '#fff' }}>{activeAnalysis.model_architecture}</b></span>
                </div>
              )}
            </div>

            {/* Session Draft Status Bar */}
            {sessionData?.status === 'draft' && (
              <div style={{
                background: 'rgba(245, 158, 11, 0.12)',
                borderTop: '1px solid rgba(245, 158, 11, 0.3)',
                padding: '10px 16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 8
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f59e0b', fontSize: 13, fontWeight: 600 }}>
                  <AlertCircle size={16} />
                  <span>DRAFT SESSION: Review findings below before saving to patient care records.</span>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    onClick={handleDiscardSession}
                    style={{
                      padding: '6px 12px',
                      borderRadius: 6,
                      background: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      color: '#ef4444',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    Discard Draft
                  </button>
                  <button
                    onClick={handleSaveSession}
                    disabled={isAnalyzing}
                    style={{
                      padding: '6px 16px',
                      borderRadius: 6,
                      background: '#00d4b8',
                      border: 'none',
                      color: '#041017',
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: 'pointer'
                    }}
                  >
                    Save & Integrate Analysis
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Visual Characteristics & Quantitative Measurements Bar */}
          {activeAnalysis && (
            <div style={{
              background: 'var(--card-bg, #0c131d)',
              border: '1px solid var(--border-color, #1a2636)',
              borderRadius: 12,
              padding: 14
            }}>
              <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-secondary, #94a3b8)', marginBottom: 10 }}>
                {activeAnalysis.analysis_type === 'surgical_wound' ? 'Quantitative Surgical Wound Biomarkers' : 'Radiographic Fracture Biomarkers'}
              </div>

              {activeAnalysis.analysis_type === 'surgical_wound' && activeAnalysis.wound_observation ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Wound Area</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.wound_area_cm2 || 'N/A'} <span style={{ fontSize: 11, fontWeight: 400 }}>cm²</span>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.wound_area_pixels} px ({activeAnalysis.wound_observation.relative_wound_size_pct}% view)
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Erythema Index</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: activeAnalysis.wound_observation.erythema_score > 50 ? '#ef4444' : '#10b981', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.erythema_score.toFixed(1)} <span style={{ fontSize: 11, fontWeight: 400 }}>/ 100</span>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.erythema_score > 50 ? 'Significant Redness' : 'Mild / Normal'}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Dehiscence Risk</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: activeAnalysis.wound_observation.dehiscence_risk_score > 50 ? '#ef4444' : '#38bdf8', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.dehiscence_risk_score.toFixed(1)} <span style={{ fontSize: 11, fontWeight: 400 }}>/ 100</span>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      Margin Separation
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Tissue Composition</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#00d4b8', textTransform: 'capitalize', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.predominant_tissue_type}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {activeAnalysis.wound_observation.exudate_detected ? `Exudate: ${activeAnalysis.wound_observation.exudate_type}` : 'No Exudate'}
                    </div>
                  </div>
                </div>
              ) : activeAnalysis.analysis_type === 'bone_fracture_xray' && activeAnalysis.fracture_analysis ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Fracture Status</div>
                    <div style={{
                      fontSize: 15,
                      fontWeight: 700,
                      color: activeAnalysis.fracture_analysis.fracture_detected ? '#ef4444' : '#10b981',
                      marginTop: 2
                    }}>
                      {activeAnalysis.fracture_analysis.fracture_detected ? 'POSITIVE' : 'NEGATIVE'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {(activeAnalysis.fracture_analysis.fracture_probability * 100).toFixed(1)}% model probability
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Anatomical Bone</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)', marginTop: 2 }}>
                      {activeAnalysis.fracture_analysis.anatomical_bone}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      Type: {activeAnalysis.fracture_analysis.fracture_type || 'N/A'}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Displacement</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: activeAnalysis.fracture_analysis.displacement_present ? '#f59e0b' : '#10b981', textTransform: 'capitalize', marginTop: 2 }}>
                      {activeAnalysis.fracture_analysis.displacement_severity}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {activeAnalysis.fracture_analysis.displacement_present ? 'Cortical step-off' : 'Anatomical alignment'}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 8, border: '1px solid var(--border-color, #1a2636)' }}>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>Callus Formation</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: activeAnalysis.fracture_analysis.callus_formation_observed ? '#10b981' : 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      {activeAnalysis.fracture_analysis.callus_formation_observed ? 'Observed' : 'Not Visualized'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)', marginTop: 2 }}>
                      Stage: {activeAnalysis.fracture_analysis.healing_stage.replace('_', ' ')}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          )}

        </div>

        {/* ------------------------------------------------------------------ */}
        {/* RIGHT PANEL: Telemetry, Gemini Reassessment & Doctor Review Hub   */}
        {/* ------------------------------------------------------------------ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Model Telemetry Card */}
          {activeAnalysis && (
            <div style={{
              background: 'var(--card-bg, #0c131d)',
              border: '1px solid var(--border-color, #1a2636)',
              borderRadius: 12,
              padding: 14
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-secondary, #94a3b8)', marginBottom: 8 }}>
                Deep Learning Model Telemetry
              </div>

              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)', marginBottom: 2 }}>
                {activeAnalysis.primary_prediction}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', marginBottom: 10 }}>
                {activeAnalysis.model_name} &bull; {activeAnalysis.model_version}
              </div>

              {/* Confidence Progress Bar */}
              <div style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary, #94a3b8)', marginBottom: 4 }}>
                  <span>Confidence Score</span>
                  <span style={{ fontWeight: 700, color: '#00d4b8' }}>
                    {(activeAnalysis.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.round(activeAnalysis.confidence_score * 100)}%`,
                    height: '100%',
                    background: '#00d4b8',
                    borderRadius: 3
                  }} />
                </div>
              </div>

              {/* Severity & Weights Badges */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 10 }}>
                <span style={{
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: activeAnalysis.severity_level === 'critical' ? 'rgba(239, 68, 68, 0.2)' : (
                    activeAnalysis.severity_level === 'elevated' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)'
                  ),
                  color: activeAnalysis.severity_level === 'critical' ? '#ef4444' : (
                    activeAnalysis.severity_level === 'elevated' ? '#f59e0b' : '#10b981'
                  ),
                  textTransform: 'uppercase'
                }}>
                  Severity: {activeAnalysis.severity_level}
                </span>

                <span style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.05)',
                  color: 'var(--text-secondary, #94a3b8)'
                }}>
                  {activeAnalysis.weights_provenance.slice(0, 24)}...
                </span>
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', lineHeight: 1.4, background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 6 }}>
                {activeAnalysis.findings_summary}
              </div>

              {/* Differential Diagnosis / Class Probabilities Breakdown */}
              {activeAnalysis.probabilities && Object.keys(activeAnalysis.probabilities).length > 1 && (
                <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-secondary, #94a3b8)', marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
                    <span>Differential Probabilities</span>
                    <span style={{ color: '#00d4b8' }}>Top Predictions</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {Object.entries(activeAnalysis.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 4)
                      .map(([className, prob]) => (
                        <div key={className} style={{ fontSize: 11 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span style={{ color: 'var(--text-primary, #f1f5f9)', fontWeight: 500 }}>{className}</span>
                            <span style={{ color: '#00d4b8', fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.round(prob * 100)}%`,
                              height: '100%',
                              background: 'linear-gradient(90deg, #00d4b8, #38bdf8)',
                              borderRadius: 2
                            }} />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Google Gemini Reassessment Card */}
          {activeAnalysis?.gemini_reassessment && (
            <div style={{
              background: 'var(--card-bg, #0c131d)',
              border: '1px solid rgba(0, 212, 184, 0.3)',
              borderRadius: 12,
              padding: 14
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                <Sparkles size={14} color="#00d4b8" />
                <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: '#00d4b8' }}>
                  Gemini API Clinical Reassessment
                </span>
              </div>

              <div style={{ fontSize: 12, color: 'var(--text-primary, #f1f5f9)', lineHeight: 1.4, marginBottom: 8 }}>
                {activeAnalysis.gemini_reassessment.reassessment_summary}
              </div>

              {activeAnalysis.gemini_reassessment.changes_vs_previous && (
                <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', marginBottom: 8 }}>
                  <b style={{ color: 'var(--text-primary, #f1f5f9)' }}>Changes vs Baseline:</b> {activeAnalysis.gemini_reassessment.changes_vs_previous}
                </div>
              )}

              {activeAnalysis.gemini_reassessment.potential_concerns?.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', marginBottom: 2 }}>Potential Concerns:</div>
                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>
                    {activeAnalysis.gemini_reassessment.potential_concerns.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                </div>
              )}

              {activeAnalysis.gemini_reassessment.recovery_plan_review_recommended && (
                <div style={{
                  padding: 8,
                  borderRadius: 6,
                  background: 'rgba(245, 158, 11, 0.12)',
                  border: '1px solid rgba(245, 158, 11, 0.25)',
                  fontSize: 11,
                  color: '#f59e0b'
                }}>
                  <b>Recovery Plan Revision Flagged:</b> {activeAnalysis.gemini_reassessment.recovery_plan_proposed_notes || 'Clinician review required.'}
                </div>
              )}
            </div>
          )}

          {/* Streamlined Doctor Review & Sign-Off Hub */}
          {activeAnalysis && (
            <div style={{
              background: 'var(--card-bg, #0c131d)',
              border: '1px solid var(--border-color, #1a2636)',
              borderRadius: 12,
              padding: 12
            }}>
              {/* If already reviewed and not in edit mode */}
              {activeAnalysis.review_status && activeAnalysis.review_status !== 'pending_review' && !editReviewMode ? (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <CheckCircle2 size={15} color={activeAnalysis.review_status === 'approved' ? '#10b981' : (activeAnalysis.review_status === 'rejected' ? '#ef4444' : '#f59e0b')} />
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)' }}>
                        Doctor Sign-Off:{' '}
                        <span style={{
                          textTransform: 'capitalize',
                          color: activeAnalysis.review_status === 'approved' ? '#10b981' : (activeAnalysis.review_status === 'rejected' ? '#ef4444' : '#f59e0b')
                        }}>
                          {activeAnalysis.review_status}
                        </span>
                      </span>
                    </div>
                    {(isDoctor || isAdmin) && (
                      <button
                        type="button"
                        onClick={() => setEditReviewMode(true)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#00d4b8',
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          padding: '2px 6px',
                          textDecoration: 'underline'
                        }}
                      >
                        Edit
                      </button>
                    )}
                  </div>
                  {activeAnalysis.clinician_notes && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', background: 'rgba(0,0,0,0.25)', padding: '6px 8px', borderRadius: 6, marginTop: 4 }}>
                      "{activeAnalysis.clinician_notes}"
                    </div>
                  )}
                  <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                    {activeAnalysis.reviewer?.name ? `Signed by Dr. ${activeAnalysis.reviewer.name}` : 'Signed into patient medical record'}
                    {activeAnalysis.is_patient_visible ? ' • Visible to patient' : ''}
                  </div>
                </div>
              ) : (
                /* Simple, streamlined review form */
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Stethoscope size={14} color="#00d4b8" />
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)' }}>
                        Doctor Sign-Off
                      </span>
                    </div>
                    {/* Status Segmented Buttons */}
                    <div style={{ display: 'flex', gap: 4 }}>
                      {[
                        { key: 'approved', label: 'Approve', color: '#10b981', bg: 'rgba(16, 185, 129, 0.2)' },
                        { key: 'amended', label: 'Amend', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.2)' },
                        { key: 'rejected', label: 'Reject', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.2)' }
                      ].map(st => (
                        <button
                          key={st.key}
                          type="button"
                          disabled={!isDoctor && !isAdmin}
                          onClick={() => {
                            setReviewStatus(st.key);
                            if (st.key === 'approved') setClinicianAgreed(true);
                            else if (st.key === 'rejected') setClinicianAgreed(false);
                          }}
                          style={{
                            padding: '3px 8px',
                            borderRadius: 6,
                            fontSize: 11,
                            fontWeight: 600,
                            cursor: (!isDoctor && !isAdmin) ? 'not-allowed' : 'pointer',
                            background: reviewStatus === st.key ? st.bg : 'rgba(0,0,0,0.2)',
                            color: reviewStatus === st.key ? st.color : 'var(--text-secondary, #94a3b8)',
                            border: `1px solid ${reviewStatus === st.key ? st.color : 'var(--border-color, #223348)'}`,
                            transition: 'all 0.15s ease'
                          }}
                        >
                          {st.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Compact Clinical Notes */}
                  <div style={{ marginBottom: 8 }}>
                    <textarea
                      rows={2}
                      value={clinicianNotes}
                      disabled={!isDoctor && !isAdmin}
                      onChange={(e) => setClinicianNotes(e.target.value)}
                      placeholder="Doctor review notes (optional)..."
                      style={{
                        width: '100%',
                        padding: '6px 8px',
                        borderRadius: 6,
                        background: 'rgba(0,0,0,0.25)',
                        border: '1px solid var(--border-color, #223348)',
                        color: 'var(--text-primary, #f1f5f9)',
                        fontSize: 11,
                        outline: 'none',
                        resize: 'none',
                        boxSizing: 'border-box'
                      }}
                    />
                  </div>

                  {/* Inline Option & Cancel (if editing) */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={isPatientVisible}
                        disabled={!isDoctor && !isAdmin}
                        onChange={(e) => setIsPatientVisible(e.target.checked)}
                        style={{ accentColor: '#00d4b8' }}
                      />
                      <span>Visible to patient in portal</span>
                    </label>

                    {editReviewMode && (
                      <button
                        type="button"
                        onClick={() => setEditReviewMode(false)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-secondary, #94a3b8)',
                          fontSize: 11,
                          cursor: 'pointer'
                        }}
                      >
                        Cancel
                      </button>
                    )}
                  </div>

                  {/* Clean Action Button */}
                  {(isDoctor || isAdmin) ? (
                    <button
                      onClick={handleReviewSubmit}
                      disabled={submittingReview}
                      style={{
                        width: '100%',
                        padding: '8px',
                        borderRadius: 6,
                        background: reviewStatus === 'rejected' ? '#ef4444' : (reviewStatus === 'amended' ? '#f59e0b' : '#00d4b8'),
                        border: 'none',
                        color: reviewStatus === 'amended' || reviewStatus === 'approved' ? '#041017' : '#ffffff',
                        fontWeight: 700,
                        fontSize: 12,
                        cursor: submittingReview ? 'not-allowed' : 'pointer'
                      }}
                    >
                      {submittingReview ? 'Signing...' : (reviewStatus === 'approved' ? '✓ Approve & Sign Off' : `Sign Off (${reviewStatus.toUpperCase()})`)}
                    </button>
                  ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)', fontStyle: 'italic', textAlign: 'center' }}>
                      Restricted to treating physicians for formal sign-off.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

        </div>

      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Research Benchmarks Modal                                          */}
      {/* ------------------------------------------------------------------ */}
      {benchmarksOpen && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.75)',
          backdropFilter: 'blur(6px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: 20
        }}>
          <div style={{
            background: '#0c131d',
            border: '1px solid var(--border-color, #223348)',
            borderRadius: 14,
            width: '90%',
            maxWidth: 1000,
            maxHeight: '85vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '16px 20px',
              borderBottom: '1px solid var(--border-color, #1a2636)'
            }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary, #f1f5f9)' }}>
                  Deep Learning Research Benchmarks & Model Evaluation
                </h3>
                <p style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--text-secondary, #94a3b8)' }}>
                  Patient-level partitioned suites across SurgWound, FUSeg, and FracAtlas datasets
                </p>
              </div>
              <button
                onClick={() => setBenchmarksOpen(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: 20, overflowY: 'auto' }}>
              {loadingBenchmarks ? (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary, #94a3b8)' }}>
                  Loading research benchmarks...
                </div>
              ) : benchmarkRuns.length === 0 ? (
                <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-secondary, #94a3b8)' }}>
                  No benchmark runs recorded.
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid var(--border-color, #1a2636)' }}>
                      <th style={{ textAlign: 'left', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Architecture</th>
                      <th style={{ textAlign: 'left', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Pipeline</th>
                      <th style={{ textAlign: 'left', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Dataset</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Accuracy</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>F1-Score</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>ROC-AUC</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Sensitivity</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Latency</th>
                      <th style={{ textAlign: 'center', padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkRuns.map((r) => (
                      <tr key={r.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 600, color: 'var(--text-primary, #f1f5f9)' }}>
                          {r.architecture}
                        </td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>
                          {r.pipeline_type.replace('_', ' ').toUpperCase()}
                        </td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-secondary, #94a3b8)' }}>
                          {r.dataset_name} ({r.total_samples} samples)
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#10b981' }}>
                          {(r.accuracy * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center', color: '#00d4b8', fontWeight: 600 }}>
                          {(r.f1_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center', color: 'var(--text-primary, #f1f5f9)' }}>
                          {(r.roc_auc * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center', color: 'var(--text-primary, #f1f5f9)' }}>
                          {(r.sensitivity * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center', color: 'var(--text-secondary, #94a3b8)' }}>
                          {r.inference_latency_ms} ms
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                          {r.is_production_model ? (
                            <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10, background: 'rgba(0, 212, 184, 0.15)', color: '#00d4b8' }}>
                              PRODUCTION
                            </span>
                          ) : (
                            <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 10, background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary, #94a3b8)' }}>
                              BENCHMARK
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
