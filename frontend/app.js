'use strict';

const DEFAULT_API_BASE = 'http://localhost:8000';
const IS_LOCAL_PAGE = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
const API_BASE = (
  localStorage.getItem('industrialFailureApiBase') ||
  (IS_LOCAL_PAGE ? DEFAULT_API_BASE : window.INDUSTRIAL_FAILURE_API_BASE) ||
  DEFAULT_API_BASE
).replace(/\/$/, '');

const els = {
  status: document.getElementById('api-status'),
  modeToggle: document.getElementById('mode-toggle'),
  knownRun: document.getElementById('known-run'),
  manualRun: document.getElementById('manual-run'),
  runLog: document.getElementById('run-log'),
  probabilityValue: document.getElementById('probability-value'),
  verdictStamp: document.getElementById('verdict-stamp'),
  costLine: document.getElementById('cost-line'),
  probabilityScale: document.getElementById('probability-scale'),
  truthLine: document.getElementById('truth-line'),
  shapTable: document.getElementById('shap-table'),
  errorBlock: document.getElementById('error-block'),
  compactWarmup: document.getElementById('compact-warmup'),
  fullWarmup: document.getElementById('full-warmup'),
  sweepPanel: document.getElementById('sweep-panel'),
  sweepFallback: document.getElementById('sweep-fallback'),
  costCurve: document.getElementById('cost-curve'),
  fnCost: document.getElementById('fn-cost'),
  fpCost: document.getElementById('fp-cost'),
  sweepOptimum: document.getElementById('sweep-optimum'),
};

const inputIds = {
  air_temperature_k: 'air_temp',
  process_temperature_k: 'proc_temp',
  rotational_speed_rpm: 'rpm',
  torque_nm: 'torque',
  tool_wear_min: 'wear',
};

const state = {
  health: null,
  riskZones: null,
  threshold: 0.775,
  probability: null,
  records: [],
  activeRecord: null,
  sweepRows: [],
};

const truthLines = {
  failCaught: "The record's outcome · this machine did fail · caught",
  failMissed:
    "The record's outcome · this machine did fail · missed. A 91.2% recall means about 9 in 100 failures slip past; this is one.",
  passAlarm: "The record's outcome · this machine did not fail · false alarm, the $2,000 kind the threshold is tuned to tolerate",
  passCorrect: "The record's outcome · this machine did not fail · correct pass",
  manual: 'your values · no recorded outcome to compare',
};

const featureGlosses = {
  power: 'mechanical power, torque times speed',
  temp_diff: 'process minus air temperature',
  wear_rate: 'wear per unit speed',
  torque_per_wear: 'torque relative to wear',
  high_load: 'high-load flag',
};

function init() {
  setupMode();
  renderScale();
  renderEmptyShap();
  bindEvents();
  loadRecords();
  loadSweep();
  checkHealth();
}

function bindEvents() {
  els.knownRun.addEventListener('click', runKnownRecord);
  els.manualRun.addEventListener('click', runManualValues);
  els.fnCost.addEventListener('input', renderCostCurve);
  els.fpCost.addEventListener('input', renderCostCurve);
  document.addEventListener('pointerdown', retireWarmups, { passive: true });
  document.addEventListener('keydown', retireWarmups);
}

function setupMode() {
  const current = document.documentElement.getAttribute('data-theme') || 'day';
  applyMode(current);
  els.modeToggle.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'night' ? 'day' : 'night';
    applyMode(next);
    persistMode(next);
  });
}

function applyMode(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  els.modeToggle.setAttribute('aria-pressed', theme === 'night' ? 'true' : 'false');
  els.modeToggle.setAttribute('aria-label', theme === 'night' ? 'Switch to light mode' : 'Switch to dark mode');
}

function persistMode(theme) {
  const mode = theme === 'night' ? 'dark' : 'light';
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `mode=${mode}; Max-Age=${maxAge}; Path=/; Domain=.alvinalias.com; SameSite=Lax`;
  try {
    localStorage.setItem('mode', mode);
  } catch (error) {
    return undefined;
  }
  return undefined;
}

async function loadRecords() {
  try {
    const response = await fetch('AA-06_records.json');
    const payload = await response.json();
    state.records = payload.records || [];
  } catch (error) {
    appendLog('> record file unavailable');
    els.knownRun.disabled = true;
  }
}

async function loadSweep() {
  try {
    const response = await fetch('sweep.json');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.sweepRows = Array.isArray(payload) ? payload : payload.rows || [];
    els.sweepPanel.classList.remove('hidden');
    els.sweepFallback.classList.add('hidden');
    renderCostCurve();
  } catch (error) {
    els.sweepPanel.classList.add('hidden');
    els.sweepFallback.classList.remove('hidden');
  }
}

async function checkHealth() {
  const warmup = beginWarmup(els.compactWarmup, 'compact');
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    state.health = data;
    state.threshold = Number(data.optimal_threshold || state.threshold);
    state.riskZones = data.risk_zones || null;
    renderScale();

    if (data.model_loaded) {
      els.status.textContent = `ready · ${data.model_type} · threshold ${formatThreshold(data.optimal_threshold)}`;
      warmup.finish();
    } else {
      els.status.textContent = 'model not loaded';
      warmup.cancel();
    }
  } catch (error) {
    els.status.textContent = 'server unreachable right now';
    warmup.cancel();
  }
}

function nextRecord() {
  if (!state.records.length) {
    return null;
  }
  const key = 'industrialFailureRecordIndex';
  const raw = Number.parseInt(localStorage.getItem(key) || '0', 10);
  const index = Number.isFinite(raw) ? raw : 0;
  const record = state.records[index % state.records.length];
  localStorage.setItem(key, String((index + 1) % state.records.length));
  return record;
}

async function runKnownRecord() {
  const record = nextRecord();
  if (!record) {
    renderError('Prediction failed. record file unavailable', 'AA-06_records.json could not be read');
    return;
  }
  state.activeRecord = record;
  setInputs(record);
  clearError();
  clearTruth();
  setLog([`> record ${record.udi} loaded · outcome hidden`]);
  await runPrediction('known');
}

async function runManualValues() {
  state.activeRecord = null;
  clearError();
  clearTruth();
  setLog([truthLines.manual]);
  await runPrediction('manual');
}

async function runPrediction(mode) {
  const payload = getPayload();
  if (!payload) {
    renderError('Prediction failed. input values are incomplete', 'Each operating point needs a number');
    return;
  }

  setBusy(true);
  const started = performance.now();
  const warmup = beginWarmup(els.fullWarmup, 'full');

  try {
    const response = await fetch(`${API_BASE}/predict?shap=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    warmup.finish();
    const seconds = ((performance.now() - started) / 1000).toFixed(2);
    appendLog(`> predicted in ${seconds} s`);
    renderResult(data, mode);
  } catch (error) {
    warmup.cancel();
    renderNetworkOrPredictionError(error);
  } finally {
    setBusy(false);
  }
}

function getPayload() {
  const payload = {};
  for (const [field, id] of Object.entries(inputIds)) {
    const value = Number.parseFloat(document.getElementById(id).value);
    if (!Number.isFinite(value)) {
      return null;
    }
    payload[field] = value;
  }
  return payload;
}

function setInputs(record) {
  document.getElementById('air_temp').value = record.air_temperature_k;
  document.getElementById('proc_temp').value = record.process_temperature_k;
  document.getElementById('rpm').value = record.rotational_speed_rpm;
  document.getElementById('torque').value = record.torque_nm;
  document.getElementById('wear').value = record.tool_wear_min;
}

function setBusy(isBusy) {
  els.knownRun.disabled = isBusy;
  els.manualRun.disabled = isBusy;
}

function setLog(lines) {
  els.runLog.innerHTML = '';
  lines.forEach(appendLog);
}

function appendLog(line) {
  const node = document.createElement('div');
  node.textContent = line;
  els.runLog.appendChild(node);
}

function renderResult(data, mode) {
  state.probability = Number(data.failure_probability);
  state.threshold = Number(data.threshold_used || state.threshold);

  const probabilityText = state.probability.toFixed(3);
  els.probabilityValue.textContent = probabilityText;
  els.probabilityValue.title = 'API field: failure_probability';

  const high = Number(data.prediction) === 1;
  els.verdictStamp.textContent = high ? 'Failure predicted' : 'No failure expected';
  els.verdictStamp.classList.toggle('is-high', high);

  if (data.estimated_cost_if_ignored !== null && data.estimated_cost_if_ignored !== undefined) {
    els.costLine.textContent = `estimated cost if ignored: $${Number(data.estimated_cost_if_ignored).toLocaleString('en-US')}`;
    els.costLine.classList.remove('hidden');
  } else {
    els.costLine.textContent = '';
    els.costLine.classList.add('hidden');
  }

  renderScale();
  renderTruthLine(data, mode);
  renderShap(data.top_shap_factors || []);
}

function renderTruthLine(data, mode) {
  els.truthLine.classList.remove('hidden', 'is-missed');
  if (mode === 'manual' || !state.activeRecord) {
    els.truthLine.textContent = truthLines.manual;
    return;
  }

  const failed = Number(state.activeRecord.machine_failure) === 1;
  const predicted = Number(data.prediction) === 1;
  if (failed && predicted) {
    els.truthLine.textContent = truthLines.failCaught;
  } else if (failed && !predicted) {
    els.truthLine.textContent = truthLines.failMissed;
    els.truthLine.classList.add('is-missed');
  } else if (!failed && predicted) {
    els.truthLine.textContent = truthLines.passAlarm;
  } else {
    els.truthLine.textContent = truthLines.passCorrect;
  }
}

function clearTruth() {
  els.truthLine.textContent = '';
  els.truthLine.classList.add('hidden');
  els.truthLine.classList.remove('is-missed');
}

function renderScale() {
  const minX = 42;
  const maxX = 958;
  const width = maxX - minX;
  const baselineY = 70;
  const threshold = clamp01(state.threshold);
  const probability = state.probability === null ? null : clamp01(state.probability);
  const ticks = [];

  for (let i = 0; i <= 20; i += 1) {
    const value = i * 0.05;
    const x = minX + value * width;
    const major = i % 5 === 0;
    ticks.push(`<line class="${major ? 'scale-tick' : 'scale-minor'}" x1="${x}" y1="${baselineY}" x2="${x}" y2="${major ? baselineY + 20 : baselineY + 12}" />`);
    if (major) {
      ticks.push(`<text class="scale-label" x="${x}" y="${baselineY + 36}" text-anchor="middle">${formatTick(value)}</text>`);
    }
  }

  const zoneLabels = [];
  if (state.riskZones) {
    const low = Number(state.riskZones.low_below);
    const high = Number(state.riskZones.high_at);
    if (Number.isFinite(low)) {
      const x = minX + clamp01(low) * width;
      zoneLabels.push(`<text class="scale-zone-label" x="${x}" y="18" text-anchor="middle">low below ${formatThreshold(low)}</text>`);
    }
    if (Number.isFinite(high)) {
      const x = minX + clamp01(high) * width;
      zoneLabels.push(`<text class="scale-zone-label" x="${x}" y="18" text-anchor="middle">high at ${formatThreshold(high)}</text>`);
    }
  }

  const thresholdX = minX + threshold * width;
  const marker = probability === null
    ? ''
    : markerTriangle(minX + probability * width, baselineY - 20, probability >= threshold);

  els.probabilityScale.innerHTML = `
    <svg viewBox="0 0 1000 124" role="img" aria-label="Failure probability from 0 to 1">
      <line class="scale-line" x1="${minX}" y1="${baselineY}" x2="${maxX}" y2="${baselineY}" />
      ${ticks.join('')}
      ${zoneLabels.join('')}
      <line class="scale-limit" x1="${thresholdX}" y1="${baselineY - 24}" x2="${thresholdX}" y2="${baselineY + 24}" />
      <text class="scale-label" x="${thresholdX + 6}" y="${baselineY - 30}">act above ${formatThreshold(threshold)}</text>
      ${marker}
    </svg>
  `;
}

function markerTriangle(x, y, isHigh) {
  return `<polygon class="scale-marker ${isHigh ? 'is-high' : ''}" points="${x},${y + 18} ${x - 8},${y} ${x + 8},${y}" />`;
}

function renderEmptyShap() {
  els.shapTable.innerHTML = '';
}

function renderShap(factors) {
  if (!factors.length) {
    renderEmptyShap();
    return;
  }
  const maxAbs = Math.max(...factors.map((factor) => Math.abs(Number(factor.shap_value))), 0.001);
  els.shapTable.innerHTML = factors.map((factor) => {
    const value = Number(factor.shap_value);
    const width = Math.max(2, Math.abs(value) / maxAbs * 50);
    const left = value >= 0 ? 50 : 50 - width;
    const sign = value > 0 ? '+' : '';
    const feature = featureLabel(factor.feature);
    const gloss = featureGloss(factor.feature);
    return `
      <div class="shap-row">
        <div class="shap-name" title="${escapeHtml(factor.feature)}">${feature}<small>${gloss}</small></div>
        <div class="shap-bar" aria-hidden="true">
          <span class="shap-zero"></span>
          <span class="shap-fill ${value < 0 ? 'negative' : 'positive'}" style="left:${left}%; width:${width}%;"></span>
        </div>
        <div class="shap-value mono">${sign}${value.toFixed(3)}</div>
      </div>
    `;
  }).join('');
}

function featureLabel(feature) {
  return String(feature)
    .replace(/\s*\[[^\]]+\]/g, '')
    .replace(/_/g, ' ')
    .toLowerCase();
}

function featureGloss(feature) {
  return featureGlosses[feature] || featureLabel(feature);
}

function renderCostCurve() {
  if (!state.sweepRows.length) {
    return;
  }

  const fnCost = Math.max(0, Number.parseFloat(els.fnCost.value) || 0);
  const fpCost = Math.max(0, Number.parseFloat(els.fpCost.value) || 0);
  const points = state.sweepRows.map((row) => ({
    t: Number(row.t),
    cost: Number(row.fn) * fnCost + Number(row.fp) * fpCost,
  }));
  let best = points[0];
  for (const point of points) {
    if (point.cost < best.cost) {
      best = point;
    }
  }

  const minCost = Math.min(...points.map((point) => point.cost));
  const maxCost = Math.max(...points.map((point) => point.cost));
  const minX = 42;
  const maxX = 958;
  const minY = 28;
  const maxY = 150;
  const costSpan = Math.max(1, maxCost - minCost);
  const xFor = (t) => minX + ((t - 0.05) / 0.9) * (maxX - minX);
  const yFor = (cost) => maxY - ((cost - minCost) / costSpan) * (maxY - minY);
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(point.t).toFixed(2)} ${yFor(point.cost).toFixed(2)}`).join(' ');
  const bestX = xFor(best.t);
  const bestY = yFor(best.cost);

  els.costCurve.innerHTML = `
    <svg viewBox="0 0 1000 190" role="img" aria-label="Cost curve across thresholds">
      <line class="curve-grid" x1="${minX}" y1="${maxY}" x2="${maxX}" y2="${maxY}" />
      <line class="curve-grid" x1="${minX}" y1="${minY}" x2="${minX}" y2="${maxY}" />
      <path class="curve-line" d="${path}" />
      <circle class="curve-marker" cx="${bestX}" cy="${bestY}" r="6" />
      <text class="curve-label" x="${bestX + 10}" y="${Math.max(18, bestY - 10)}">best ${formatThreshold(best.t)}</text>
      <text class="curve-label" x="${minX}" y="176">0.05</text>
      <text class="curve-label" x="${maxX}" y="176" text-anchor="end">0.95</text>
    </svg>
  `;
  els.sweepOptimum.textContent = `at these costs the best threshold is ${formatThreshold(best.t)}`;
}

function renderNetworkOrPredictionError(error) {
  const raw = error && error.message ? error.message : String(error);
  if (raw === 'Failed to fetch' || raw.includes('NetworkError')) {
    renderError(
      'Prediction failed. This ML demo sleeps after extended inactivity; the Space may still be waking. Try again in a moment.',
      raw,
    );
  } else {
    renderError(`Prediction failed. ${raw}`, raw);
  }
}

function renderError(message, detail) {
  els.errorBlock.classList.remove('hidden');
  els.errorBlock.innerHTML = `
    <div>${escapeHtml(message)}</div>
    <div class="error-detail mono">${escapeHtml(detail)}</div>
  `;
}

function clearError() {
  els.errorBlock.classList.add('hidden');
  els.errorBlock.innerHTML = '';
}

function beginWarmup(container, mode) {
  const started = performance.now();
  let visible = false;
  let overrunLogged = false;
  let intervalId = null;
  let timeoutId = null;
  let done = false;

  const draw = (ready = false) => {
    const elapsed = (performance.now() - started) / 1000;
    const remaining = Math.max(0, Math.ceil(60 - elapsed));
    const overrun = !ready && elapsed > 60;
    if (overrun && !overrunLogged) {
      overrunLogged = true;
    }
    const value = ready ? 0 : overrun ? Math.floor(elapsed) : remaining;
    const label = ready ? 'ready' : overrun ? 'seconds elapsed · still starting' : 'estimated seconds to warm';
    const log = ready
      ? `> awake · measured wake time ${elapsed.toFixed(1)} s`
      : overrun
        ? '> past the usual window · still waiting, counting up honestly'
        : '> warm-up estimate counting · this is an estimate, not progress';
    container.innerHTML = warmupMarkup({
      marker: ready || overrun ? 0 : remaining,
      value,
      label,
      log,
      overrun,
      mode,
    });
  };

  timeoutId = window.setTimeout(() => {
    if (done) {
      return;
    }
    visible = true;
    container.classList.remove('hidden');
    if (mode === 'compact') {
      els.status.textContent = 'model waking, first run may take up to a minute';
    }
    appendLogOnce('> server was asleep · sent the wake call');
    draw();
    intervalId = window.setInterval(draw, 1000);
  }, 2500);

  return {
    finish() {
      done = true;
      window.clearTimeout(timeoutId);
      window.clearInterval(intervalId);
      if (visible) {
        draw(true);
        window.setTimeout(() => container.classList.add('hidden'), prefersReducedMotion() ? 0 : 4000);
      }
    },
    cancel() {
      done = true;
      window.clearTimeout(timeoutId);
      window.clearInterval(intervalId);
      container.classList.add('hidden');
    },
  };
}

function warmupMarkup({ marker, value, label, log, overrun }) {
  const minX = 24;
  const maxX = 376;
  const y = 34;
  const x = minX + (Math.max(0, Math.min(60, marker)) / 60) * (maxX - minX);
  const ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60].map((tick) => {
    const tx = minX + (tick / 60) * (maxX - minX);
    const major = tick % 15 === 0;
    return `
      <line class="warm-tick" x1="${tx}" y1="${y}" x2="${tx}" y2="${major ? y + 16 : y + 9}" />
      ${major ? `<text class="warm-label" x="${tx}" y="${y + 32}" text-anchor="middle">${tick}</text>` : ''}
    `;
  }).join('');

  return `
    <div class="warmup-meter">
      <svg viewBox="0 0 400 76" role="img" aria-label="Warm-up estimate">
        <line class="warm-line" x1="${minX}" y1="${y}" x2="${maxX}" y2="${y}" />
        ${ticks}
        <polygon class="scale-marker ${overrun ? 'is-high' : ''}" points="${x},${y - 14} ${x - 8},${y - 2} ${x + 8},${y - 2}" />
      </svg>
      <div class="warm-number ${overrun ? 'is-overrun' : ''}">
        <span class="warm-value">${value}</span>
        <span class="warm-caption">${label}</span>
      </div>
      <div class="warm-log">${log}</div>
    </div>
  `;
}

function appendLogOnce(line) {
  const exists = Array.from(els.runLog.children).some((node) => node.textContent === line);
  if (!exists) {
    appendLog(line);
  }
}

function retireWarmups() {
  els.compactWarmup.classList.add('hidden');
  if (!els.fullWarmup.textContent.includes('still waiting')) {
    els.fullWarmup.classList.add('hidden');
  }
}

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function formatTick(value) {
  if (value === 0 || value === 1) {
    return String(value);
  }
  return value.toFixed(2).replace(/0$/, '');
}

function formatThreshold(value) {
  return Number(value).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

init();
