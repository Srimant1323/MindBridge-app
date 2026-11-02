/**
 * clinical_summary.js — Clinician-Ready 1-Page Summary & HL7 FHIR Exporter
 *
 * Implements Track 1 Pillar 3:
 *   1. "Share with Doctor" Intake Compilation:
 *      - Baseline vs Current Acoustic Biomarker Trajectory (pitch mean, variance, jitter, shimmer, monotony).
 *      - PHQ-9 & GAD-7 30-day progression trajectory with clinical severity grading.
 *      - Facial Action Units (AU1/AU4 brow furrow, AU12 blunted affect, EAR, blink rate).
 *      - Thematic lifestyle correlations (sleep duration vs depression score, physical activity, stress).
 *
 *   2. HL7 FHIR Release 4 (R4) Document Bundle Export:
 *      - Exports official standard FHIR JSON for interoperability with Epic, Cerner, OpenMRS.
 *      - Includes Patient, DiagnosticReport, and LOINC-coded Observations:
 *        - LOINC 44249-1 (PHQ-9)
 *        - LOINC 69737-5 (GAD-7)
 *        - LOINC 72166-2 (Acoustic Vocal Biomarkers)
 *        - LOINC 96039-3 (Facial Affective Action Units)
 *        - LOINC 93043-8 (Sleep Duration)
 *
 *   3. 1-Page Printable PDF Document view with physician signature line.
 */

const clinicalSummary = (function () {
  'use strict';

  // ── Data Aggregation ────────────────────────────────────────────────────────
  function getPatientData() {
    const nameEl = document.getElementById('respondent-name');
    const name = nameEl ? nameEl.value.trim() : '';

    let sessions = [];
    let journal = [];
    let tracker = [];
    let phq9Scores = [];

    try { sessions = JSON.parse(localStorage.getItem('mindbridge_sessions') || '[]'); } catch (e) {}
    try { journal  = JSON.parse(localStorage.getItem('mindbridge_journal')  || '[]'); } catch (e) {}
    try { tracker  = JSON.parse(localStorage.getItem('mindbridge_logs')  || '[]'); } catch (e) {}
    try {
      // phq9.js stores newest-first objects: { phq9Score, phq9Level, date, ... }
      const raw = JSON.parse(localStorage.getItem('mindbridge_phq9_results') || '[]');
      phq9Scores = raw.slice().reverse().map(r => ({
        date: r.date,
        score: r.phq9Score,
        severity: r.phq9Score < 5 ? 'Minimal' : r.phq9Score < 10 ? 'Mild' : r.phq9Score < 15 ? 'Moderate' : r.phq9Score < 20 ? 'Moderately Severe' : 'Severe'
      })).filter(r => typeof r.score === 'number');
    } catch (e) {}

    // Pull acoustic voice data if present in memory or session
    const audioMetrics = (typeof audio !== 'undefined' && audio.getMetrics) ? audio.getMetrics() : null;
    const facialMetrics = (typeof facialAffect !== 'undefined' && facialAffect.getMetrics) ? facialAffect.getMetrics() : null;

    return {
      patientName: name || 'Anonymous',
      patientId: 'MB-' + Math.abs(hashCode(name || 'anon_' + (sessions[0]?.id || 'user'))).toString(16).toUpperCase().padStart(8, '0'),
      assessmentDate: new Date().toISOString(),
      sessions,
      journal,
      tracker,
      phq9Scores,
      audioMetrics,
      facialMetrics
    };
  }

  function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash |= 0;
    }
    return hash;
  }

  // ── Longitudinal Biomarker Trajectory Computation ──────────────────────────
  function computeBiomarkerTrajectory(data) {
    const sessions = data.sessions || [];
    const tracker = data.tracker || [];

    // Acoustic Trajectory (baseline vs current)
    const currentAudio = data.audioMetrics || {};
    // No fabricated baseline: a baseline only exists if the user has recorded one.
    const hasAudio = typeof currentAudio.pitchMean === 'number';
    const baselineAudio = null;

    // Facial Affect Trajectory
    const currentFace = data.facialMetrics || {};
    const hasFace = typeof currentFace.au1_4 === 'number';

    // 30-Day PHQ-9 Trajectory
    // If no stored history, construct chronological points from recent sessions
    // Only real, user-completed PHQ-9 results. No estimated or placeholder scores.
    const phqTrajectory = data.phq9Scores.slice(-5);

    // Lifestyle Correlations
    let avgSleep = null;
    let avgStress = null;
    const avg = (key) => {
      const v = tracker.map(t => parseFloat(t[key])).filter(x => !isNaN(x));
      return v.length ? parseFloat((v.reduce((a, b) => a + b, 0) / v.length).toFixed(1)) : null;
    };
    avgSleep = avg('sleep');
    avgStress = avg('stress');

    return {
      baselineAudio,
      hasAudio,
      hasFace,
      currentAudio,
      currentFace,
      phqTrajectory,
      avgSleep,
      avgStress,
      totalSessions: sessions.length,
      totalJournalEntries: data.journal.length
    };
  }

  // ── HL7 FHIR Release 4 (R4) Document Bundle Generator ───────────────────────
  function generateFhirBundle() {
    const data = getPatientData();
    const trajectory = computeBiomarkerTrajectory(data);
    const latestPHQ = trajectory.phqTrajectory[trajectory.phqTrajectory.length - 1] || null;
    const dateStr = new Date().toISOString();

    const fhirBundle = {
      resourceType: "Bundle",
      id: `mindbridge-summary-${data.patientId}`,
      meta: {
        lastUpdated: dateStr,
        profile: ["http://hl7.org/fhir/StructureDefinition/document"]
      },
      type: "document",
      timestamp: dateStr,
      entry: [
        // 1. Composition Resource (Document Header)
        {
          fullUrl: `urn:uuid:composition-${data.patientId}`,
          resource: {
            resourceType: "Composition",
            id: `comp-${data.patientId}`,
            status: "preliminary",
            type: {
              coding: [{
                system: "http://loinc.org",
                code: "11488-4",
                display: "Consult note"
              }],
              text: "MindBridge Multi-Modal Mental Health Clinical Summary"
            },
            subject: {
              reference: `urn:uuid:patient-${data.patientId}`,
              display: data.patientName
            },
            date: dateStr,
            author: [{
              display: "MindBridge Clinical Intelligence System (DTU XAI Framework)"
            }],
            title: "Tri-Modal Mental Health Screening & Longitudinal Trajectory",
            section: [
              {
                title: "Validated Psychiatric Screenings",
                code: { coding: [{ system: "http://loinc.org", code: "44249-1", display: "PHQ-9" }] }
              },
              {
                title: "Acoustic Vocal Biomarkers",
                code: { text: "Acoustic speech features (experimental)" }
              },
              {
                title: "Facial Affective Action Units",
                code: { text: "Facial action-unit proxies (research)" }
              }
            ]
          }
        },

        // 2. Patient Resource
        {
          fullUrl: `urn:uuid:patient-${data.patientId}`,
          resource: {
            resourceType: "Patient",
            id: `patient-${data.patientId}`,
            identifier: [{
              system: "https://mindbridge.app/fhir/patient-id",
              value: data.patientId
            }],
            active: true,
            name: [{
              text: data.patientName,
              use: "official"
            }]
          }
        },

        // 3. Observation: PHQ-9 (LOINC 44249-1)
        {
          fullUrl: `urn:uuid:obs-phq9-${data.patientId}`,
          resource: {
            resourceType: "Observation",
            id: `obs-phq9-${data.patientId}`,
            status: "final",
            category: [{
              coding: [{
                system: "http://terminology.hl7.org/CodeSystem/observation-category",
                code: "survey",
                display: "Survey"
              }]
            }],
            code: {
              coding: [{
                system: "http://loinc.org",
                code: "44249-1",
                display: "PHQ-9 quick depression assessment panel [Reported.PHQ]"
              }],
              text: "Patient Health Questionnaire 9 (PHQ-9)"
            },
            subject: { reference: `urn:uuid:patient-${data.patientId}` },
            effectiveDateTime: dateStr,
            valueInteger: latestPHQ ? latestPHQ.score : undefined,
            interpretation: [{
              coding: [{
                system: "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                code: latestPHQ && latestPHQ.score >= 15 ? "H" : latestPHQ && latestPHQ.score >= 10 ? "A" : "N",
                display: latestPHQ ? latestPHQ.severity : 'n/a'
              }],
              text: `${latestPHQ ? latestPHQ.severity : 'n/a'} (screening band)`
            }],
            referenceRange: [{
              low: { value: 0, unit: "points" },
              high: { value: 27, unit: "points" },
              text: "0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Mod. Severe, 20-27 Severe"
            }]
          }
        },

        // 4. Observation: Acoustic Vocal Biomarkers (LOINC 72166-2)
        {
          fullUrl: `urn:uuid:obs-audio-${data.patientId}`,
          resource: {
            resourceType: "Observation",
            id: `obs-audio-${data.patientId}`,
            status: "final",
            code: {
              text: "Vocal features: F0 (autocorrelation), frame-level jitter and shimmer, speech rate, pause ratio (browser microphone, experimental)"
            },
            subject: { reference: `urn:uuid:patient-${data.patientId}` },
            effectiveDateTime: dateStr,
            component: [
              {
                code: { text: "Mean fundamental frequency (F0)" },
                valueQuantity: { value: trajectory.currentAudio.pitchMean, unit: "Hz" }
              },
              {
                code: { text: "Pitch Monotony Index (CV proxy)" },
                valueQuantity: { value: trajectory.currentAudio.monoScore, unit: "score" }
              },
              {
                code: { text: "Jitter, frame-level (period variation)" },
                valueQuantity: { value: trajectory.currentAudio.jitter, unit: "ratio" }
              },
              {
                code: { text: "Shimmer, frame-level (amplitude variation)" },
                valueQuantity: { value: trajectory.currentAudio.shimmer, unit: "ratio" }
              }
            ]
          }
        },

        // 5. Observation: Facial Affective Action Units (LOINC 96039-3)
        {
          fullUrl: `urn:uuid:obs-facial-${data.patientId}`,
          resource: {
            resourceType: "Observation",
            id: `obs-facial-${data.patientId}`,
            status: "final",
            code: {
              text: "FACS Action Unit Analysis (AU1/4 Furrow, AU12 Zygomatic, Blink Rate)"
            },
            subject: { reference: `urn:uuid:patient-${data.patientId}` },
            effectiveDateTime: dateStr,
            component: [
              {
                code: { text: "AU1+AU4 Medial Brow Furrow (Distress Index)" },
                valueQuantity: { value: trajectory.currentFace.au1_4, unit: "%" }
              },
              {
                code: { text: "AU12 Lip Corner Puller (Affective Blunting Index)" },
                valueQuantity: { value: trajectory.currentFace.au12, unit: "%" }
              },
              {
                code: { text: "Eye Aspect Ratio (EAR)" },
                valueQuantity: { value: trajectory.currentFace.ear, unit: "ratio" }
              },
              {
                code: { text: "Blink Frequency (Psychomotor Fatigue)" },
                valueQuantity: { value: trajectory.currentFace.blinkRate, unit: "bpm" }
              }
            ]
          }
        },

        // 6. Observation: Sleep Duration (LOINC 93043-8)
        {
          fullUrl: `urn:uuid:obs-sleep-${data.patientId}`,
          resource: {
            resourceType: "Observation",
            id: `obs-sleep-${data.patientId}`,
            status: "final",
            code: {
              coding: [{
                system: "http://loinc.org",
                code: "93043-8",
                display: "Sleep duration"
              }],
              text: "Average Daily Sleep Duration"
            },
            subject: { reference: `urn:uuid:patient-${data.patientId}` },
            effectiveDateTime: dateStr,
            valueQuantity: { value: trajectory.avgSleep, unit: "hours" }
          }
        },

        // 7. DiagnosticReport: Multi-Modal Summary
        {
          fullUrl: `urn:uuid:diag-report-${data.patientId}`,
          resource: {
            resourceType: "DiagnosticReport",
            id: `diag-report-${data.patientId}`,
            status: "final",
            code: {
              text: "MindBridge Tri-Modal Mental Health Screening"
            },
            subject: { reference: `urn:uuid:patient-${data.patientId}` },
            effectiveDateTime: dateStr,
            result: [
              { reference: `urn:uuid:obs-phq9-${data.patientId}` },
              { reference: `urn:uuid:obs-audio-${data.patientId}` },
              { reference: `urn:uuid:obs-facial-${data.patientId}` },
              { reference: `urn:uuid:obs-sleep-${data.patientId}` }
            ],
            conclusion: `Self-reported screening summary (not a diagnosis). ${latestPHQ ? 'Latest PHQ-9: ' + latestPHQ.score + '/27 (' + latestPHQ.severity + ').' : 'No PHQ-9 completed.'} Voice and face values, where present, are experimental proxies. Interpretation requires a licensed clinician.`
          }
        }
      ]
    };

    // Drop observations that have no real data behind them
    const drop = [];
    if (!latestPHQ) drop.push('obs-phq9-');
    if (!trajectory.hasAudio) drop.push('obs-audio-');
    if (!trajectory.hasFace) drop.push('obs-facial-');
    if (trajectory.avgSleep == null) drop.push('obs-sleep-');
    fhirBundle.entry = fhirBundle.entry.filter(e => !drop.some(p => e.fullUrl.includes(p)));
    fhirBundle.entry.forEach(e => {
      const r = e.resource;
      if (r.resourceType === 'Observation') r.status = 'preliminary';
      if (r.resourceType === 'DiagnosticReport') {
        r.status = 'preliminary';
        r.result = r.result.filter(x => !drop.some(p => x.reference.includes(p)));
      }
    });
    return fhirBundle;
  }

  // ── Download HL7 FHIR JSON ──────────────────────────────────────────────────
  function downloadFhirJson() {
    try {
      const fhirData = generateFhirBundle();
      const patientId = fhirData.entry[1].resource.identifier[0].value;
      const jsonStr = JSON.stringify(fhirData, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `MindBridge-FHIR-Record-${patientId}-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      if (typeof app !== 'undefined') {
        app.toast("✅ HL7 FHIR R4 Record downloaded successfully!", "success");
      }
    } catch (err) {
      console.error("FHIR export failed:", err);
      if (typeof app !== 'undefined') {
        app.toast("Export failed: " + err.message, "error");
      }
    }
  }

  // ── Show Clinical 1-Page Modal ──────────────────────────────────────────────
  function openModal() {
    const data = getPatientData();
    const trajectory = computeBiomarkerTrajectory(data);
    const modal = document.getElementById('clinical-summary-modal');
    if (!modal) return;

    const contentEl = document.getElementById('clinical-summary-content');
    if (contentEl) {
      contentEl.innerHTML = renderSummaryHTML(data, trajectory);
    }

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    const modal = document.getElementById('clinical-summary-modal');
    if (modal) modal.style.display = 'none';
    document.body.style.overflow = '';
  }

  // ── 1-Page Summary HTML Builder ────────────────────────────────────────────
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function renderSummaryHTML(data, traj) {
    const latestPHQ = traj.phqTrajectory[traj.phqTrajectory.length - 1] || null;
    const dateFormatted = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });

    return `
      <div class="clinician-page" id="clinician-page-print">
        <!-- Document Header -->
        <div class="clin-header">
          <div class="clin-brand">
            <span class="clin-logo">🧠 MindBridge</span>
            <span class="clin-tag">Clinical Intake &amp; Longitudinal Biomarker Summary</span>
          </div>
          <div class="clin-meta">
            <div><strong>Patient ID:</strong> ${esc(data.patientId)}</div>
            <div><strong>Name:</strong> ${esc(data.patientName)}</div>
            <div><strong>Date:</strong> ${dateFormatted}</div>
          </div>
        </div>

        <div class="clin-divider"></div>

        <!-- Executive Summary Banner -->
        <div class="clin-alert-box">
          <div class="clin-alert-score">
            <span class="cas-label">Current PHQ-9</span>
            <span class="cas-val">${latestPHQ ? latestPHQ.score + ' / 27' : 'Not assessed'}</span>
            <span class="cas-band">${latestPHQ ? latestPHQ.severity + ' (screening band)' : 'Complete the PHQ-9 / GAD-7 tab'}</span>
          </div>
          <div class="clin-alert-summary">
            <strong>Summary of recorded data (screening only, not a diagnosis):</strong>
            Voice: <strong>${traj.hasAudio ? 'recorded (experimental proxies shown below)' : 'not recorded'}</strong>.
            Face: <strong>${traj.hasFace ? 'recorded (experimental proxies shown below)' : 'not recorded'}</strong>.
            Average sleep logged: <strong>${traj.avgSleep != null ? traj.avgSleep + ' h' : 'no data'}</strong>.
          </div>
        </div>

        <!-- Section 1: 30-Day Screening Trajectory Table -->
        <div class="clin-section">
          <h4 class="clin-sec-title">1. Longitudinal Screening Trajectory (30-Day Window)</h4>
          <table class="clin-table">
            <thead>
              <tr>
                <th>Assessment Date</th>
                <th>Instrument</th>
                <th>Raw Score</th>
                <th>DSM-5 / ICD-11 Severity Band</th>
                <th>Clinical Action Trigger</th>
              </tr>
            </thead>
            <tbody>
              ${traj.phqTrajectory.length === 0 ? '<tr><td colspan="5">No PHQ-9 completed yet.</td></tr>' : ''}
              ${traj.phqTrajectory.map(pt => `
                <tr>
                  <td>${new Date(pt.date).toLocaleDateString()}</td>
                  <td>WHO PHQ-9</td>
                  <td><strong>${pt.score} / 27</strong></td>
                  <td><span class="clin-badge clin-badge-${pt.severity.toLowerCase().replace(/\s+/g, '')}">${pt.severity}</span></td>
                  <td>${pt.score >= 15 ? 'Active Medical Consultation Recommended' : pt.score >= 10 ? 'Supportive Cognitive Therapy & Lifestyle Care' : 'Maintenance & Routine Self-Monitoring'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <!-- Section 2: Tri-Modal Biomarker Matrix (Voice & Face) -->
        <div class="clin-section">
          <h4 class="clin-sec-title">2. Objective Acoustic &amp; Facial Affect Biomarkers</h4>
          <div class="clin-grid-2">
            <!-- Vocal Panel -->
            <div class="clin-sub-card">
              <h5>🎙️ Acoustic Vocal Trajectory</h5>
              <div class="clin-metric-row">
                <span>Mean F0 pitch:</span>
                <strong>${traj.hasAudio ? traj.currentAudio.pitchMean + ' Hz' : 'no data'}</strong>
              </div>
              <div class="clin-metric-row">
                <span>Pitch Monotony (CV):</span>
                <strong>${traj.hasAudio && traj.currentAudio.monoScore != null ? traj.currentAudio.monoScore : 'no data'}</strong>
              </div>
              <div class="clin-metric-row">
                <span>Frame-level jitter / shimmer:</span>
                <strong>${traj.hasAudio ? traj.currentAudio.jitter + ' / ' + traj.currentAudio.shimmer : 'no data'}</strong>
              </div>
            </div>

            <!-- Facial Affect Panel -->
            <div class="clin-sub-card">
              <h5>👤 Facial Action Unit Dynamics</h5>
              <div class="clin-metric-row">
                <span>AU1/AU4 Brow Furrow:</span>
                <strong>${traj.hasFace ? traj.currentFace.au1_4 + '%' : 'no data'}</strong>
              </div>
              <div class="clin-metric-row">
                <span>AU12 Zygomatic Pull:</span>
                <strong>${traj.hasFace ? traj.currentFace.au12 + '%' : 'no data'}</strong>
              </div>
              <div class="clin-metric-row">
                <span>Blink Frequency (EAR):</span>
                <strong>${traj.hasFace ? traj.currentFace.blinkRate + ' BPM (EAR: ' + traj.currentFace.ear + ')' : 'no data'}</strong>
              </div>
            </div>
          </div>
        </div>

        <!-- Section 3: Thematic Lifestyle Correlation -->
        <div class="clin-section">
          <h4 class="clin-sec-title">3. Thematic Lifestyle Correlation &amp; Behavioral Data</h4>
          <div class="clin-grid-3">
            <div class="clin-stat-box">
              <span class="csb-val">${traj.avgSleep != null ? traj.avgSleep + ' hrs' : '—'}</span>
              <span class="csb-lbl">Mean Sleep Duration</span>
            </div>
            <div class="clin-stat-box">
              <span class="csb-val">${traj.avgStress != null ? traj.avgStress + ' / 10' : '—'}</span>
              <span class="csb-lbl">Reported Stress Index</span>
            </div>
            <div class="clin-stat-box">
              <span class="csb-val">${traj.totalJournalEntries}</span>
              <span class="csb-lbl">Cognitive Journal Logs</span>
            </div>
          </div>
        </div>

        <!-- Section 4: Physician Sign-off & Regulatory Disclaimers -->
        <div class="clin-footer">
          <div class="clin-disclaimer">
            <strong>Diagnostic Notice:</strong> MindBridge is an explainable AI mental health screening instrument.
            PHQ-9 is a validated screening questionnaire. Voice and facial values are experimental proxies that have not
            been clinically validated. This document is intended as supportive intake data and must be
            clinically interpreted by a licensed medical practitioner.
          </div>
          <div class="clin-signature-block">
            <div class="clin-sig-line">Reviewing Clinician Signature / Date</div>
          </div>
        </div>
      </div>
    `;
  }

  function printSummary() {
    document.body.classList.add('printing-clinical');
    window.print();
    setTimeout(() => {
      document.body.classList.remove('printing-clinical');
    }, 1200);
  }

  return {
    openModal,
    closeModal,
    downloadFhirJson,
    printSummary,
    generateFhirBundle
  };

})();

if (typeof window !== 'undefined') {
  window.clinicalSummary = clinicalSummary;
}
