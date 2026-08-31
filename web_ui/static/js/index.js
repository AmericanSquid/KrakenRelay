// Reverse-proxy subdirectory support (/goodsam, /stjoes, etc.)
const MOUNT_PREFIX = (() => {
  const seg = window.location.pathname.split('/').filter(Boolean)[0];
  return seg ? `/${seg}` : '';
})();

function apiUrl(path){
  const p = String(path || '').replace(/^\/+/, '');
  return MOUNT_PREFIX ? `${MOUNT_PREFIX}/${p}` : `/${p}`;
}

// ====== Polling visibility guard ======
function shouldPoll() {
  return document.visibilityState === "visible";
}

// ====== Tabs ======
function setupTabGroup(groupName, storageKey, fallbackId){
  const buttons = Array.from(document.querySelectorAll(`[data-tab-group="${groupName}"][data-tab-btn]`));
  const panels = Array.from(document.querySelectorAll(`[data-tab-group="${groupName}"][data-tab-panel]`));

  function resolveTabId(tabId){
    const direct = panels.find(panel => panel.id === tabId);
    const directButton = buttons.find(btn => btn.dataset.tabBtn === tabId);
    if (direct && !(directButton && directButton.hidden)) return tabId;

    const fallback = panels.find(panel => panel.id === fallbackId);
    const fallbackButton = buttons.find(btn => btn.dataset.tabBtn === fallbackId);
    if (fallback && !(fallbackButton && fallbackButton.hidden)) return fallback.id;

    const firstVisibleButton = buttons.find(btn => !btn.hidden);
    if (firstVisibleButton) return firstVisibleButton.dataset.tabBtn;

    return panels[0] ? panels[0].id : null;
  }

  function activate(tabId, persist = true){
    const actualId = resolveTabId(tabId);
    if (!actualId) return;

    buttons.forEach(btn => {
      const active = btn.dataset.tabBtn === actualId;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });

    panels.forEach(panel => {
      const active = panel.id === actualId;
      panel.classList.toggle('is-active', active);
      panel.hidden = !active;
    });

    if (persist){
      try { localStorage.setItem(storageKey, actualId); } catch (_) {}
    }
  }

  buttons.forEach(btn => {
    btn.addEventListener('click', () => activate(btn.dataset.tabBtn));
  });

  let initial = fallbackId;
  try {
    initial = localStorage.getItem(storageKey) || initial;
  } catch (_) {}

  const current = buttons.find(btn => btn.classList.contains('is-active'))?.dataset.tabBtn;
  activate(current || initial, false);

  return {
    activate,
    current(){
      return panels.find(panel => panel.classList.contains('is-active'))?.id || null;
    },
    restore(){
      let saved = fallbackId;
      try { saved = localStorage.getItem(storageKey) || saved; } catch (_) {}
      activate(saved, false);
    }
  };
}

const mainTabController = setupTabGroup('main', 'krakenrelay-tab-main', 'tab-basic-setup');
const repeaterTabController = setupTabGroup('repeater', 'krakenrelay-tab-repeater', 'repeater-cw');
const dspTabController = setupTabGroup('dsp', 'krakenrelay-tab-dsp', 'dsp-basic');

const expertToggle = document.getElementById('expert-toggle');

function setAdvancedVisibility(show){
  document.querySelectorAll('.advanced-only').forEach(el => {
    el.hidden = !show;
  });

  if (!show) {
    if (mainTabController.current() === 'tab-advanced') {
      mainTabController.activate('tab-basic-setup', false);
    } else {
      mainTabController.activate(mainTabController.current() || 'tab-basic-setup', false);
    }

    if (dspTabController.current() === 'dsp-advanced') {
      dspTabController.activate('dsp-basic', false);
    } else {
      dspTabController.activate(dspTabController.current() || 'dsp-basic', false);
    }
  } else {
    mainTabController.restore();
    dspTabController.restore();
  }

  try { localStorage.setItem('krakenrelay-advanced', show ? '1' : '0'); } catch (_) {}

  if (typeof syncConditionalDisables === 'function') {
    syncConditionalDisables();
  }
}

try {
  expertToggle.checked = localStorage.getItem('krakenrelay-advanced') === '1';
} catch (_) {
  expertToggle.checked = false;
}

// ====== UI Readouts (sliders) ======
function bindTextReadout(inputId, outputId, formatter){
  const input = document.getElementById(inputId);
  const output = document.getElementById(outputId);
  if (!input || !output) return;

  const render = () => {
    output.textContent = formatter(input.value);
  };

  input.addEventListener('input', render);
  render();
}

bindTextReadout('squelch', 'squelch-value', value => `${value} dBFS`);
bindTextReadout('highpass-cutoff', 'highpass-value', value => `${value} Hz`);
bindTextReadout('limiter-threshold', 'limiter-value', value => Number(value).toFixed(3));
bindTextReadout('courtesy-vol', 'courtesy-vol-value', value => value);
bindTextReadout('cw-vol', 'cw-vol-value', value => value);
bindTextReadout('tot-vol', 'tot-vol-value', value => value);
bindTextReadout('compressor-strength', 'compressor-strength-value', value => `${value}%`);
bindTextReadout('speex-suppression', 'speex-suppression-value', value => `${value} dB`);

function compressorMacro(percent){
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  const s = p / 100.0;
  return {
    threshold: -15.0 - (10.0 * s),
    ratio: 1.8 + (2.4 * s),
    makeup: 2.5 + (2.5 * s)
  };
}

const compressorStrengthEl = document.getElementById('compressor-strength');
const compressorThresholdEl = document.getElementById('compressor-threshold');
const compressorRatioEl = document.getElementById('compressor-ratio');
const compressorMakeupEl = document.getElementById('compressor-makeup');

function syncCompressorMacroFields(pushUpdates){
  if (!compressorStrengthEl || !compressorThresholdEl || !compressorRatioEl || !compressorMakeupEl) return;

  const macro = compressorMacro(compressorStrengthEl.value);
  compressorThresholdEl.value = macro.threshold.toFixed(1);
  compressorRatioEl.value = macro.ratio.toFixed(1);
  compressorMakeupEl.value = macro.makeup.toFixed(1);

  if (pushUpdates && !document.getElementById('cfg-lock').checked) {
    [compressorThresholdEl, compressorRatioEl, compressorMakeupEl].forEach(scheduleUpdate);
  }
}

if (compressorStrengthEl) {
  compressorStrengthEl.addEventListener('input', () => syncCompressorMacroFields(false));
  compressorStrengthEl.addEventListener('change', () => syncCompressorMacroFields(true));
}

const toneEnabledEl = document.getElementById('tx-tone-enabled');
const toneRangeEl = document.getElementById('tx-tone');
const toneKnobEl = document.getElementById('tx-tone-knob');
const toneValueEl = document.getElementById('tx-tone-value');

function setToneValue(nextValue, emitChange = false){
  if (!toneRangeEl) return;
  const min = Number(toneRangeEl.min || -1);
  const max = Number(toneRangeEl.max || 1);
  const clamped = Math.max(min, Math.min(max, nextValue));
  toneRangeEl.value = clamped.toFixed(2);
  toneRangeEl.dispatchEvent(new Event('input', { bubbles: true }));
  if (emitChange) {
    toneRangeEl.dispatchEvent(new Event('change', { bubbles: true }));
  }
}

function renderToneKnob(){
  if (!toneRangeEl || !toneKnobEl || !toneValueEl) return;
  const value = Number(toneRangeEl.value || 0);
  const angle = -135 + ((value + 1) / 2) * 270;
  toneKnobEl.style.setProperty('--knob-angle', `${angle}deg`);
  toneKnobEl.setAttribute('aria-valuenow', value.toFixed(2));
  toneKnobEl.classList.toggle('is-disabled', !!toneRangeEl.disabled);
  toneValueEl.textContent = value.toFixed(2);
}

if (toneRangeEl && toneKnobEl) {
  toneRangeEl.addEventListener('input', renderToneKnob);

  toneKnobEl.addEventListener('wheel', event => {
    if (toneRangeEl.disabled || document.getElementById('cfg-lock').checked) return;
    event.preventDefault();
    const step = Number(toneRangeEl.step || 0.05);
    const direction = event.deltaY < 0 ? 1 : -1;
    setToneValue(Number(toneRangeEl.value || 0) + (direction * step), true);
  }, { passive: false });

  toneKnobEl.addEventListener('keydown', event => {
    if (toneRangeEl.disabled || document.getElementById('cfg-lock').checked) return;
    const step = Number(toneRangeEl.step || 0.05);
    if (event.key === 'ArrowUp' || event.key === 'ArrowRight') {
      event.preventDefault();
      setToneValue(Number(toneRangeEl.value || 0) + step, true);
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowLeft') {
      event.preventDefault();
      setToneValue(Number(toneRangeEl.value || 0) - step, true);
    }
  });

  toneKnobEl.addEventListener('pointerdown', event => {
    if (toneRangeEl.disabled || document.getElementById('cfg-lock').checked) return;
    event.preventDefault();

    const startY = event.clientY;
    const startValue = Number(toneRangeEl.value || 0);

    const move = moveEvent => {
      const delta = (startY - moveEvent.clientY) * 0.01;
      setToneValue(startValue + delta, false);
    };

    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      toneRangeEl.dispatchEvent(new Event('change', { bubbles: true }));
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });

  renderToneKnob();
}

// ====== Helpers ======
function setDot(el, on){
  if(on){ el.classList.add('on'); }
  else { el.classList.remove('on'); }
}

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

function setRunPillState(running, rx, tx){
  const runPill = document.getElementById('run-pill');
  runPill.classList.remove('state-stopped', 'state-running', 'state-rx', 'state-tx');
  if (!running) runPill.classList.add('state-stopped');
  else if (tx) runPill.classList.add('state-tx');
  else if (rx) runPill.classList.add('state-rx');
  else runPill.classList.add('state-running');
}

// ====== Start/Stop ======
document.getElementById('start-btn').addEventListener('click', () => {
  const formData = new FormData(document.getElementById('config-form'));
  fetch(apiUrl('start'), { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'running') {
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
      } else {
        alert('Failed to start: ' + (data.message || 'Unknown error'));
      }
    })
    .catch(err => alert('Start failed: ' + err));
});

document.getElementById('stop-btn').addEventListener('click', () => {
  fetch(apiUrl('stop'), { method: 'POST' })
    .then(res => res.json())
    .then(() => {
      document.getElementById('start-btn').disabled = false;
      document.getElementById('stop-btn').disabled = true;
    });
});

// These endpoints aren’t in your web_ui.py yet — leaving buttons in place,
// but we won’t crash the UI if they 404.
document.getElementById('id-btn').addEventListener('click', () => {
  fetch(apiUrl('manual_id'), { method: 'POST' }).catch(()=>{});
});


// ====== Recording Plugin Controls ======
const recordStartBtn = document.getElementById('record-start-btn');
const recordStopBtn = document.getElementById('record-stop-btn');
const recordingPill = document.getElementById('recording-pill');

function setRecordingUI(recording, file){
  if (!recordStartBtn || !recordStopBtn || !recordingPill) return;

  recordStartBtn.disabled = recording || !uiRunning;
  recordStopBtn.disabled = !recording || !uiRunning;

  recordingPill.classList.remove('state-stopped', 'state-running', 'state-rx', 'state-tx');

  if (recording) {
    recordingPill.classList.add('state-tx');
    recordingPill.textContent = 'REC: On';
    recordingPill.title = file || '';
  } else {
    recordingPill.classList.add('state-stopped');
    recordingPill.textContent = 'REC: Off';
    recordingPill.title = '';
  }
}

async function updateRecordingStatus(){
  if (!shouldPoll()) return;

  if (!uiRunning) {
    setRecordingUI(false, null);
    return;
  }

  try {
    const res = await fetch(apiUrl('plugins/recording/status'), { cache: "no-store" });
    const status = await res.json();

    setRecordingUI(!!status.recording, status.file || null);
  } catch (_) {
    setRecordingUI(false, null);
  }
}

if (recordStartBtn) {
  recordStartBtn.addEventListener('click', async () => {
    try {
      const res = await fetch(apiUrl('plugins/recording/start'), {
        method: 'POST',
        cache: 'no-store'
      });

      const result = await res.json();

      if (!res.ok || result.ok === false) {
        alert('Recording failed: ' + (result.error || result.message || 'Unknown error'));
        return;
      }

      setRecordingUI(true, result.file || null);
      updateRecordingStatus();
    } catch (err) {
      alert('Recording start failed: ' + err);
    }
  });
}

if (recordStopBtn) {
  recordStopBtn.addEventListener('click', async () => {
    try {
      const res = await fetch(apiUrl('plugins/recording/stop'), {
        method: 'POST',
        cache: 'no-store'
      });

      const result = await res.json();

      if (!res.ok || result.ok === false) {
        alert('Stop recording failed: ' + (result.error || result.message || 'Unknown error'));
        return;
      }

      setRecordingUI(false, null);
      updateRecordingStatus();
    } catch (err) {
      alert('Recording stop failed: ' + err);
    }
  });
}

setInterval(updateRecordingStatus, 2000);

// ====== Status polling ======
let uiRunning = false;

async function updateStatus() {
  if (!shouldPoll()) return;

  try {
    const res = await fetch(apiUrl('status'), { cache: "no-store" });
    const status = await res.json();

    uiRunning = !!status.running;

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const txDot = document.getElementById('tx-dot');
    const rxDot = document.getElementById('rx-dot');
    const runPill = document.getElementById('run-pill');
    const pttPill = document.getElementById('ptt-pill');
    const autoErr = document.getElementById('auto-start-error');

    if (status.auto_start_error) {
      autoErr.style.display = "block";
      autoErr.textContent = status.auto_start_error;
    } else {
      autoErr.style.display = "none";
    }

    if (!status.running) {
      setDot(txDot, false);
      setDot(rxDot, false);
      runPill.textContent = 'Stopped';
      setRunPillState(false, false, false);
      pttPill.textContent = 'PTT: Not Started';

      document.getElementById('tot-lockout-indicator').style.display = 'none';
      document.getElementById('id-indicator').style.display = 'none';
      document.getElementById('tot-text').textContent = '—';
      document.getElementById('next-id-text').textContent = '—';
      document.getElementById('tot-bar').style.width = '0%';

      startBtn.disabled = false;
      stopBtn.disabled = true;

      return;
    }

    startBtn.disabled = true;
    stopBtn.disabled  = false;

    // lights
    setDot(txDot, !!status.tx);
    setDot(rxDot, !!status.rx);
    runPill.textContent = status.tx ? 'Running (TX)' : (status.rx ? 'Running (RX)' : 'Running');
    setRunPillState(!!status.running, !!status.rx, !!status.tx);
    updateRecordingStatus();

    // ptt
    if (status.ptt_status_text) {
      pttPill.textContent = status.ptt_status_text;
      if (status.ptt_status_color) pttPill.style.color = status.ptt_status_color;
    }

    // TOT lockout badge
    const totBadge = document.getElementById('tot-lockout-indicator');
    totBadge.style.display = status.tot_lockout ? 'inline-flex' : 'none';

    // TOT progress
    const totEnabled = !!status.tot_enabled;
    const totElapsed = (typeof status.tot_elapsed === 'number') ? status.tot_elapsed : 0;
    const totLimit = (typeof status.tot_limit === 'number') ? status.tot_limit : 0;

    if (!totEnabled) {
      document.getElementById('tot-text').textContent = 'Disabled';
      document.getElementById('tot-bar').style.width = '0%';
    } else if (totLimit > 0) {
      document.getElementById('tot-text').textContent = `${totElapsed.toFixed(1)} / ${totLimit}s`;
      const totPct = clamp((totElapsed / totLimit) * 100, 0, 100);
      document.getElementById('tot-bar').style.width = totPct + '%';
    } else {
      document.getElementById('tot-text').textContent = `${totElapsed.toFixed(1)}s`;
      document.getElementById('tot-bar').style.width = '0%';
    }

    // ID indicator + next id time
    const idBadge = document.getElementById('id-indicator');
    idBadge.style.display = status.sending_id ? 'inline-flex' : 'none';

    if (typeof status.next_id_in === 'number') {
      document.getElementById('next-id-text').textContent = Math.ceil(status.next_id_in) + 's';
    } else {
      document.getElementById('next-id-text').textContent = '—';
    }

  } catch (_) {
    uiRunning = false;
  }
}

// Cloudflare-friendlier than 500ms
setInterval(updateStatus, 2000);
updateStatus();

// ====== Smooth meters ======
let targetRxDb = -60, shownRxDb = -60;
let targetTxDb = -60, shownTxDb = -60;

function smoothStep(shown, target){
  const attack = 0.35;   // fast rise
  const release = 0.08;  // slow fall
  const a = (target > shown) ? attack : release;
  return shown + (target - shown) * a;
}

function dbToPct(db){
  return Math.min(100, Math.max(0, (db + 60) / 60 * 100));
}

function meterTick(){
  shownRxDb = smoothStep(shownRxDb, targetRxDb);
  shownTxDb = smoothStep(shownTxDb, targetTxDb);

  document.getElementById('level-bar').style.height = dbToPct(shownRxDb) + '%';
  document.getElementById('level-text').textContent = shownRxDb.toFixed(1) + ' dBFS';

  requestAnimationFrame(meterTick);
}

requestAnimationFrame(meterTick);

async function updateMeter() {
  if (!shouldPoll()) return;

  if (!uiRunning) {
    targetRxDb = -60;
    targetTxDb = -60;
    return;
  }

  try {
    const r = await fetch(apiUrl('meter'), { cache: "no-store" });
    const m = await r.json();

    if (!m.running) {
      targetRxDb = -60;
      targetTxDb = -60;
      return;
    }

    targetRxDb = (typeof m.rx_db === 'number') ? m.rx_db : -60;
    targetTxDb = (typeof m.tx_db === 'number') ? m.tx_db : -60;

    const clip = document.getElementById('clip-alert');
    const limit = document.getElementById('limit-alert');

    if (m.clipping) clip.classList.remove('hidden'); else clip.classList.add('hidden');
    if (m.limiting) limit.classList.remove('hidden'); else limit.classList.add('hidden');
  } catch (_) {}
}

// Was 50ms. That is spicy over Cloudflare.
// 1000ms still gives useful remote visibility without request spam.
setInterval(updateMeter, 1000);

// ====== Conservative Config Lock + Live Updates + Save ======
const lockEl = document.getElementById("cfg-lock");
const lockState = document.getElementById("lock-state");
const saveBtn = document.getElementById("save-btn");
const saveStatus = document.getElementById("save-status");
const restartBadge = document.getElementById("restart-badge");

const dualInputEl = document.getElementById("dual-input");
const input2El = document.getElementById("input-device-2");
const dualOutputEl = document.getElementById("dual-output");
const output2ModeEl = document.getElementById("output-2-mode");
const out2El = document.getElementById("output-device-2");
const courtesyToneEl = document.getElementById("courtesy-tone");
const courtesyVolEl = document.getElementById("courtesy-vol");
const cwEnabledEl = document.getElementById("cw-enabled");
const totLockoutEl = document.getElementById("tot-lockout");
const lockoutTimeEl = document.getElementById("lockout-time");
const highpassEl = document.getElementById("highpass");
const highpassCutoffEl = document.getElementById("highpass-cutoff");
const limiterEnabledEl = document.getElementById("limiter-enabled");
const limiterThresholdEl = document.getElementById("limiter-threshold");
const compressorEnabledEl = document.getElementById("compressor-enabled");
const notchEnabledEl = document.getElementById("notch-enabled");
const notchFrequencyEl = document.getElementById("notch-frequency");
const notchQEl = document.getElementById("notch-q");
const notchHarmonicsEl = document.getElementById("notch-harmonics");
const speexEnabledEl = document.getElementById("speex-enabled");
const speexSuppressionEl = document.getElementById("speex-suppression");
const speexDenoiseEl = document.getElementById("speex-denoise");
const speexAgcEl = document.getElementById("speex-agc");
const speexAgcLevelEl = document.getElementById("speex-agc-level");
const speexVadEl = document.getElementById("speex-vad");
const dualPttEl = document.getElementById("dual-ptt");
const pttModePrimaryEl = document.getElementById("ptt-mode-primary");
const pttModeSecondaryEl = document.getElementById("ptt-mode-secondary");
const secondaryPttModeWrapEl = document.getElementById("secondary-ptt-mode-wrap");
const pttDevicePrimaryEl = document.getElementById("ptt-device-primary");
const pttPinPrimaryEl = document.getElementById("ptt-pin-primary");
const pttDeviceSecondaryEl = document.getElementById("ptt-device-secondary");
const pttPinSecondaryEl = document.getElementById("ptt-pin-secondary");
const secondaryPttAdvancedWrapEl = document.getElementById("secondary-ptt-advanced-wrap");

setAdvancedVisibility(!!expertToggle.checked);
expertToggle.addEventListener('change', () => setAdvancedVisibility(expertToggle.checked));

async function postJSON(url, body){
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });

  let payload = {};
  try { payload = await r.json(); } catch (_) {}

  if (!r.ok){
    const msg = payload?.message || payload?.error || payload?.status || r.statusText || "request failed";
    throw new Error(`${r.status} ${msg}`);
  }

  return payload;
}

function setSavePill(text, cls){
  saveStatus.style.display = "inline-flex";
  saveStatus.textContent = text;
  saveStatus.classList.remove("ok","warn","bad");
  if (cls) saveStatus.classList.add(cls);
  setTimeout(() => { saveStatus.style.display = "none"; }, 1500);
}

function syncConditionalDisables(){
  const locked = lockEl.checked;
  const advancedVisible = expertToggle.checked;

  if (highpassEl && highpassCutoffEl){
    highpassCutoffEl.disabled = locked || !highpassEl.checked;
  }

  if (limiterEnabledEl && limiterThresholdEl){
    limiterThresholdEl.disabled = locked || !limiterEnabledEl.checked;
  }

  if (input2El && dualInputEl){
    input2El.disabled = locked || !dualInputEl.checked;
  }

  if (out2El && dualOutputEl){
    out2El.disabled = locked || !dualOutputEl.checked;
  }

  if (output2ModeEl && dualOutputEl){
    output2ModeEl.disabled = locked || !dualOutputEl.checked;
  }

  if (courtesyVolEl && courtesyToneEl){
    courtesyVolEl.disabled = locked || !courtesyToneEl.checked;
  }

  ["cw-speed","cw-pitch","cw-vol"].forEach(id => {
    const el = document.getElementById(id);
    if (el && cwEnabledEl){
      el.disabled = locked || !cwEnabledEl.checked;
    }
  });

  if (lockoutTimeEl && totLockoutEl){
    lockoutTimeEl.disabled = locked || !totLockoutEl.checked;
  }

  if (compressorEnabledEl){
    const compressorBasic = [document.getElementById("compressor-strength")];
    compressorBasic.forEach(el => {
      if (el) el.disabled = locked || !compressorEnabledEl.checked;
    });

    [compressorThresholdEl, compressorRatioEl, compressorMakeupEl,
     document.getElementById("compressor-attack"),
     document.getElementById("compressor-release")].forEach(el => {
      if (el) el.disabled = locked || !compressorEnabledEl.checked || !advancedVisible;
    });
  }

  if (notchEnabledEl){
    if (notchFrequencyEl) notchFrequencyEl.disabled = locked || !notchEnabledEl.checked;
    if (notchQEl) notchQEl.disabled = locked || !notchEnabledEl.checked || !advancedVisible;
    if (notchHarmonicsEl) notchHarmonicsEl.disabled = locked || !notchEnabledEl.checked || !advancedVisible;
  }

  if (speexEnabledEl){
    if (speexSuppressionEl) speexSuppressionEl.disabled = locked || !speexEnabledEl.checked;
    [speexDenoiseEl, speexAgcEl, speexAgcLevelEl, speexVadEl].forEach(el => {
      if (el) el.disabled = locked || !speexEnabledEl.checked || !advancedVisible;
    });
  }

  if (toneRangeEl && toneEnabledEl){
    toneRangeEl.disabled = locked || !toneEnabledEl.checked;
    renderToneKnob();
  }

  const dualPtt = !!(dualPttEl && dualPttEl.checked);
  if (secondaryPttModeWrapEl) secondaryPttModeWrapEl.hidden = !dualPtt;
  if (secondaryPttAdvancedWrapEl) secondaryPttAdvancedWrapEl.hidden = !dualPtt || !advancedVisible;
  if (pttModeSecondaryEl) pttModeSecondaryEl.disabled = locked || !dualPtt;

  const primaryCm108 = String(pttModePrimaryEl?.value || "").toUpperCase() === "CM108";
  if (pttDevicePrimaryEl) pttDevicePrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;
  if (pttPinPrimaryEl) pttPinPrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;

  const secondaryCm108 = String(pttModeSecondaryEl?.value || "").toUpperCase() === "CM108";
  if (pttDeviceSecondaryEl) pttDeviceSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;
  if (pttPinSecondaryEl) pttPinSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;
}

function setLockedUI(locked){
  lockState.textContent = locked ? "LOCKED" : "UNLOCKED";
  lockState.style.color = locked ? "#ffcc33" : "#3dff47";

  const form = document.getElementById('config-form');
  const configCard = form ? form.closest('.card') : null;

  if (configCard){
    configCard.querySelectorAll('input, select, textarea, button').forEach(el => {
      if (el === lockEl) return;
      if (el.id === 'save-btn') return;
      if (el.id === 'expert-toggle') return;
      if (el.dataset.tabBtn) return;
      el.disabled = locked;
    });
  }

  syncConditionalDisables();
}

lockEl.checked = true;
setLockedUI(true);

fetch(apiUrl("lock"), { cache: "no-store" })
  .then(r => r.json())
  .then(d => {
    lockEl.checked = !!d.locked;
    setLockedUI(lockEl.checked);
  })
  .catch(() => {
    lockEl.checked = true;
    setLockedUI(true);
  });

lockEl.addEventListener('change', () => {
  if (!lockEl.checked) {
    const ok = window.confirm(
      "You are about to unlock configuration controls\n\n" +
      "Changing repeater settings may disrupt normal operation.\n\n" +
      "Are you sure you want to proceed?"
    );

    if (!ok) {
      lockEl.checked = true;
      setLockedUI(true);
      return;
    }

    setLockedUI(false);
    postJSON(apiUrl("lock"), { locked: false })
      .catch(err => {
        console.error("Failed to unlock backend config:", err);
        setSavePill("Backend lock failed", "bad");
        lockEl.checked = true;
        setLockedUI(true);
      });
    return;
  }

  setLockedUI(true);
  postJSON(apiUrl("lock"), { locked: true })
    .catch(err => {
      console.error("Failed to lock backend config:", err);
      setSavePill("Backend lock failed", "bad");
    });
});

const debounceTimers = new Map();

function valueForConfigElement(el){
  if (el.type === "checkbox") return el.checked;
  return el.value;
}

async function sendLiveUpdate(el){
  if (lockEl.checked) {
    return { skipped: true, reason: "locked" };
  }

  const key = el.dataset.key;
  if (!key) {
    return { skipped: true, reason: "missing-key" };
  }

  const value = valueForConfigElement(el);

  try {
    const resp = await postJSON(apiUrl("config/live"), { key, value });

    if (resp && resp.status === "restart_required"){
      restartBadge.style.display = "inline-flex";
      el.classList.add("needs-restart");
    }

    el.classList.remove("save-error");
    return resp || { status: "ok" };

  } catch (err) {
    console.error("Live config update failed:", key, err);
    el.classList.add("save-error");
    setSavePill(`Update failed: ${key}`, "bad");
    throw err;
  }
}

function scheduleUpdate(el){
  const key = el.dataset.key || el.id;

  if (debounceTimers.has(key)) {
    clearTimeout(debounceTimers.get(key));
  }

  debounceTimers.set(
    key,
    setTimeout(() => {
      debounceTimers.delete(key);
      sendLiveUpdate(el).catch(() => {});
    }, 120)
  );
}

async function flushPendingConfigUpdates(){
  // Kill pending debounce timers so Save cannot race them.
  debounceTimers.forEach(timer => clearTimeout(timer));
  debounceTimers.clear();

  if (lockEl.checked) {
    throw new Error("Unlock config first");
  }

  const elements = Array.from(document.querySelectorAll("[data-key]"))
    .filter(el => !el.disabled);

  for (const el of elements) {
    await sendLiveUpdate(el);
  }
}

document.querySelectorAll("[data-key]").forEach(el => {
  const handler = () => {
    if (lockEl.checked) return;
    scheduleUpdate(el);
    syncConditionalDisables();
  };

  if (el.tagName === "SELECT" || el.type === "checkbox") {
    el.addEventListener("change", handler);
  } else {
    el.addEventListener("input", handler);
  }
});

saveBtn.addEventListener("click", async () => {
  if (lockEl.checked) {
    setSavePill("Unlock config first", "warn");
    return;
  }

  saveBtn.disabled = true;
  setSavePill("Saving…", "warn");

  try {
    await flushPendingConfigUpdates();

    const res = await fetch(apiUrl("config/apply"), {
      method: "POST",
      cache: "no-store"
    });

    const data = await res.json().catch(() => ({}));

    if (res.ok && data && data.status === "ok") {
      setSavePill("Saved ✅", "ok");

      if (restartBadge) {
        restartBadge.style.display = "none";
      }

      document.querySelectorAll(".needs-restart").forEach(el => {
        el.classList.remove("needs-restart");
      });

    } else {
      const msg = data && (data.error || data.message || data.status)
        ? (data.error || data.message || data.status)
        : "Save failed";

      console.error("Config apply failed:", data);
      setSavePill(msg, "bad");
    }

  } catch (err) {
    console.error("Save failed:", err);
    setSavePill(err.message || "Save failed ❌", "bad");

  } finally {
    saveBtn.disabled = false;
  }
});

syncConditionalDisables();

// ====== System stats polling ======
async function updateSystemStats() {
  if (!shouldPoll()) return;

  try {
    const res = await fetch(apiUrl("api/stats"), { cache: "no-store" });
    const stats = await res.json();

    updateTempPill(stats);
    updateLoadPill(stats.load);
    updateUptimePill(stats.uptime);
    updateMumblePill(stats);
  } catch (err) {
    console.error("Failed to fetch system stats", err);
  }
}

function updateTempPill(stats) {
  const pill = document.getElementById("temp-pill");
  if (!pill) return;

  const temps = [stats.cpu_temp, stats.nvme_temp].filter(t => t !== null);

  if (temps.length === 0) {
    pill.textContent = "TEMP —";
    pill.className = "pill";
    return;
  }

  const maxTemp = Math.max(...temps);

  let cls =
    maxTemp >= 80 ? "temp-hot" :
    maxTemp >= 70 ? "temp-warn" :
                    "temp-ok";

  pill.textContent = `TEMP ${maxTemp.toFixed(0)}°C`;
  pill.className = `pill ${cls}`;

  pill.title =
    `CPU: ${stats.cpu_temp?.toFixed(1) ?? "—"}°C\n` +
    `NVMe: ${stats.nvme_temp?.toFixed(1) ?? "—"}°C`;
}

function updateLoadPill(load) {
  const pill = document.getElementById("load-pill");
  if (!pill) return;

  if (load === null) {
    pill.textContent = "LOAD —";
    pill.className = "pill";
    return;
  }

  let cls =
    load >= 2.0 ? "temp-hot" :
    load >= 1.0 ? "temp-warn" :
                  "temp-ok";

  pill.textContent = `LOAD ${load.toFixed(2)}`;
  pill.className = `pill ${cls}`;
}

function updateUptimePill(seconds) {
  const pill = document.getElementById("uptime-pill");
  if (!pill) return;
  pill.textContent = `UP ${formatUptime(seconds)}`;
}

function formatUptime(seconds) {
  if (seconds === null) return "—";

  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);

  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function updateMumblePill(stats) {
  const pill = document.getElementById("mumble-pill");
  if (!pill) return;

  // backend provides stats.mumble (boolean)
  const active = stats.mumble;

  if (active === true) {
    pill.textContent = "Mumble OK";
    pill.className = "pill temp-ok";
  } else if (active === false) {
    pill.textContent = "Mumble DOWN";
    pill.className = "pill temp-hot";
  } else {
    pill.textContent = "Mumble ?";
    pill.className = "pill unknown";
  }
}

// poll every 5s
setInterval(updateSystemStats, 5000);
updateSystemStats();

// ====== Logs polling ======
let lastLogId = 0;

async function fetchLogs() {
  if (!shouldPoll()) return;

  try {
    const res = await fetch(apiUrl(`logs?after=${lastLogId}`), { cache: "no-store" });
    const data = await res.json();

    const container = document.getElementById("logContainer");
    if (!container) return;

    const shouldAutoscroll =
      container.scrollHeight - container.scrollTop - container.clientHeight < 40;

    data.entries.forEach(entry => {
      lastLogId = entry.id;

      const line = document.createElement("div");
      line.className = `log-line log-${entry.level}`;

      const time = new Date(entry.ts * 1000).toLocaleTimeString();

      line.textContent =
        `[${time}] ${entry.level} ${entry.message}`;

      container.appendChild(line);
    });

    if (shouldAutoscroll) {
      container.scrollTop = container.scrollHeight;
    }

  } catch (err) {
    console.error("Log fetch failed:", err);
  }
}

// Was 1000ms. Logs do not need to hit Cloudflare once per second.
setInterval(fetchLogs, 10000);
fetchLogs();

async function clearLogs() {
  await fetch(apiUrl("logs/clear"), { method: "POST" });

  const container = document.getElementById("logContainer");
  if (container) container.innerHTML = "";

  lastLogId = 0;
}

// ====== Maintenance polling ======
async function updateMaintenanceStatus() {
  if (!shouldPoll()) return;

  try {
    const res = await fetch(apiUrl("maintenance"), { cache: "no-store" });
    const data = await res.json();

    const indicator = document.getElementById("maintenanceIndicator");
    if (!indicator) return;

    indicator.classList.remove("on", "off", "restarting");

    if (data.restarting) {
      indicator.classList.add("restarting");
      indicator.textContent = "Controller Restarting";
    }
    else if (data.maintenance) {
      indicator.classList.add("on");
      indicator.textContent = "System Under Maintenance";
    } else {
      indicator.classList.add("off");
      indicator.textContent = "Normal Operation";
    }

  } catch (err) {
    console.error("Maintenance status check failed:", err);
  }
}

// Was 2000ms. 5000ms is plenty for remote dashboard status.
setInterval(updateMaintenanceStatus, 5000);
updateMaintenanceStatus();

// ====== Refresh immediately when tab becomes visible again ======
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;

  updateStatus();
  updateMeter();
  updateSystemStats();
  fetchLogs();
  updateMaintenanceStatus();
});


// KR_PLUGIN_RECORDING_JS_START
// ====== Recording Plugin UI ======
const recordToggleBtn = document.getElementById('record-toggle-btn');
const recordIcon = document.getElementById('record-icon');
const recordLabel = document.getElementById('record-label');
const recordStatusText = document.getElementById('record-status-text');
const recordDuration = document.getElementById('record-duration');
const recordFile = document.getElementById('record-file');

let recordingState = {
  recording: false,
  file: null,
  started_at: null,
  elapsed_seconds: 0,
};

function formatRecordingDuration(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));

  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function setRecordingUI(status) {
  if (!recordToggleBtn || !recordIcon || !recordLabel) return;

  recordingState = {
    recording: !!status.recording,
    file: status.file || null,
    started_at: status.started_at || null,
    elapsed_seconds: Number(status.elapsed_seconds || 0),
  };

  const isRecording = recordingState.recording;
  recordToggleBtn.disabled = !uiRunning;

  recordIcon.classList.remove('record-icon-idle', 'record-icon-active');

  if (isRecording) {
    recordIcon.textContent = '■';
    recordIcon.classList.add('record-icon-active');
    recordLabel.textContent = 'Stop';

    if (recordStatusText) recordStatusText.textContent = 'Recording';
    if (recordFile) recordFile.textContent = recordingState.file || '—';
  } else {
    recordIcon.textContent = '●';
    recordIcon.classList.add('record-icon-idle');
    recordLabel.textContent = 'Record';

    if (recordStatusText) recordStatusText.textContent = uiRunning ? 'Idle' : 'Controller stopped';
    if (recordFile) recordFile.textContent = '—';
  }

  if (recordDuration) {
    recordDuration.textContent = formatRecordingDuration(recordingState.elapsed_seconds);
  }
}

async function updateRecordingStatus() {
  if (!shouldPoll()) return;

  if (!uiRunning) {
    setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
    return;
  }

  try {
    const res = await fetch(apiUrl('plugins/recording/status'), { cache: 'no-store' });
    const status = await res.json();

    if (!res.ok || status.ok === false) {
      setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
      return;
    }

    setRecordingUI(status);
  } catch (_) {
    setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
  }
}

if (recordToggleBtn) {
  recordToggleBtn.addEventListener('click', async () => {
    const action = recordingState.recording ? 'stop' : 'start';

    try {
      recordToggleBtn.disabled = true;

      const res = await fetch(apiUrl(`plugins/recording/${action}`), {
        method: 'POST',
        cache: 'no-store',
      });

      const result = await res.json();

      if (!res.ok || result.ok === false) {
        alert('Recording action failed: ' + (result.error || result.message || 'Unknown error'));
        return;
      }

      await updateRecordingStatus();
    } catch (err) {
      alert('Recording action failed: ' + err);
    } finally {
      recordToggleBtn.disabled = !uiRunning;
    }
  });
}

setInterval(updateRecordingStatus, 2000);
updateRecordingStatus();

setInterval(() => {
  if (!recordingState.recording || !recordDuration) return;

  recordingState.elapsed_seconds += 1;
  recordDuration.textContent = formatRecordingDuration(recordingState.elapsed_seconds);
}, 1000);
// KR_PLUGIN_RECORDING_JS_END

