'use strict';

const DEFAULT_API_BASE = 'http://localhost:8000';
const API_BASE = (
  window.INDUSTRIAL_FAILURE_API_BASE ||
  localStorage.getItem('industrialFailureApiBase') ||
  DEFAULT_API_BASE
).replace(/\/$/, '');

// ─── API health check ─────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById('api-status');
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    const data = await resp.json();
    if (data.model_loaded) {
      badge.textContent = `✅ API ready · ${data.model_type} · threshold ${data.optimal_threshold}`;
      badge.className = 'status-badge status-ok';
    } else {
      badge.textContent = '⚠️ API degraded — model not loaded';
      badge.className = 'status-badge status-error';
    }
  } catch {
    badge.textContent = '❌ API offline';
    badge.className = 'status-badge status-error';
  }
}

// ─── Prediction ──────────────────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = 'Predicting…';

  const payload = {
    air_temperature_k:      parseFloat(document.getElementById('air_temp').value),
    process_temperature_k:  parseFloat(document.getElementById('proc_temp').value),
    rotational_speed_rpm:   parseFloat(document.getElementById('rpm').value),
    torque_nm:              parseFloat(document.getElementById('torque').value),
    tool_wear_min:          parseFloat(document.getElementById('wear').value),
  };

  try {
    const resp = await fetch(`${API_BASE}/predict?shap=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderResult(data);
  } catch (e) {
    alert(`Prediction failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict Failure Risk';
  }
}

// ─── Render result ───────────────────────────────────────────────────────
function renderResult(data) {
  const section = document.getElementById('result-section');
  section.classList.remove('hidden');

  // Risk badge
  const badge = document.getElementById('risk-badge');
  const icons = { low: '✅', medium: '⚠️', high: '🔴' };
  const labels = { low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk' };
  badge.className = `risk-badge risk-${data.risk_level}`;
  document.getElementById('risk-icon').textContent = icons[data.risk_level] || '❓';
  document.getElementById('risk-label').textContent = labels[data.risk_level] || data.risk_level;

  // Probability
  const pct = (data.failure_probability * 100).toFixed(1);
  document.getElementById('prob-number').textContent = `${pct}%`;
  const predText = data.prediction === 1 ? 'Failure predicted' : 'No failure';
  document.getElementById('threshold-text').textContent =
    `Threshold: ${data.threshold_used} · ${predText}`;

  // Cost warning
  const costDiv = document.getElementById('cost-warning');
  if (data.estimated_cost_if_ignored) {
    costDiv.classList.remove('hidden');
    document.getElementById('cost-value').textContent =
      `$${data.estimated_cost_if_ignored.toLocaleString()}`;
  } else {
    costDiv.classList.add('hidden');
  }

  // SHAP factors
  const shapSection = document.getElementById('shap-section');
  if (data.top_shap_factors && data.top_shap_factors.length > 0) {
    shapSection.classList.remove('hidden');
    renderShapBars(data.top_shap_factors);
  } else {
    shapSection.classList.add('hidden');
  }

  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderShapBars(factors) {
  const container = document.getElementById('shap-bars');
  container.innerHTML = '';
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

  for (const f of factors) {
    const pct = Math.round((Math.abs(f.shap_value) / maxAbs) * 100);
    const dir = f.shap_value > 0 ? 'risk-up' : 'risk-down';
    const sign = f.shap_value > 0 ? '+' : '';
    const shortName = f.feature.replace(' [K]', 'K').replace(' [rpm]', ' rpm')
                               .replace(' [Nm]', ' Nm').replace(' [min]', ' min');

    container.insertAdjacentHTML('beforeend', `
      <div class="shap-row">
        <div class="shap-name" title="${f.feature}">${shortName}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${dir}" style="width:${pct}%"></div>
        </div>
        <div class="shap-value" style="color:${f.shap_value>0?'var(--red)':'var(--green)'}">
          ${sign}${f.shap_value.toFixed(3)}
        </div>
      </div>
    `);
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────
checkHealth();
