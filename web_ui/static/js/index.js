// Reverse-proxy subdirectory support (/goodsam, /stjoes, etc.)
const MOUNT_PREFIX = (() => {
  const seg = window.location.pathname.split('/').filter(Boolean)[0];
  return seg ? `/${seg}` : '';
})();

function apiUrl(path){
  const p = String(path || '').replace(/^\/+/, '');
  return MOUNT_PREFIX ? `${MOUNT_PREFIX}/${p}` : `/${p}`;
}

// ====== Tabs ======
const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
const tabPanels = Array.from(document.querySelectorAll('.config-panel'));

function activateTab(tabId){
  tabButtons.forEach(btn => {
    const active = btn.dataset.tabBtn === tabId;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });

  tabPanels.forEach(panel => {
    const active = panel.id === tabId;
    panel.classList.toggle('is-active', active);
    panel.hidden = !active;
  });

  try { localStorage.setItem('krakenrelay-active-tab', tabId); } catch (_) {}
}

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tabBtn));
});

try {
  const savedTab = localStorage.getItem('krakenrelay-active-tab');
  if (savedTab && document.getElementById(savedTab)) activateTab(savedTab);
} catch (_) {}

// ====== UI Readouts (sliders) ======
const squelchSlider = document.getElementById('squelch');
const squelchValue = document.getElementById('squelch-value');
squelchSlider.addEventListener('input', () => {
  squelchValue.textContent = squelchSlider.value + ' dB';
});

const hpToggle = document.getElementById('highpass');
const hpCutoff = document.getElementById('highpass-cutoff');
const hpValue = document.getElementById('highpass-value');
hpToggle.addEventListener('change', () => {
  hpCutoff.disabled = !hpToggle.checked || document.getElementById("cfg-lock").checked;
});
hpCutoff.addEventListener('input', () => {
  hpValue.textContent = hpCutoff.value + ' Hz';
});

const limiterThr = document.getElementById('limiter-threshold');
const limiterVal = document.getElementById('limiter-value');
limiterThr.addEventListener('input', () => {
  limiterVal.textContent = Number(limiterThr.value).toFixed(3);
});

const courtesyVol = document.getElementById('courtesy-vol');
const courtesyVolValue = document.getElementById('courtesy-vol-value');
courtesyVol.addEventListener('input', () => {
  courtesyVolValue.textContent = courtesyVol.value;
});

const cwVol = document.getElementById('cw-vol');
const cwVolValue = document.getElementById('cw-vol-value');
cwVol.addEventListener('input', () => {
  cwVolValue.textContent = cwVol.value;
});

const totVol = document.getElementById('tot-vol');
const totVolValue = document.getElementById('tot-vol-value');
totVol.addEventListener('input', () => {
  totVolValue.textContent = totVol.value;
});

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
document.getElementById('test-ptt').addEventListener('click', () => {
  fetch(apiUrl('ptt_test'), { method: 'POST' }).catch(()=>{});
});

// ====== Status polling (slow) ======
let uiRunning = false;

setInterval(() => {
  fetch(apiUrl('status'), { cache: "no-store" })
    .then(res => res.json())
    .then(status => {
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
        return;
      }

      if (status.running) {
        startBtn.disabled = true;
        stopBtn.disabled  = false;
      } else {
        startBtn.disabled = false;
        stopBtn.disabled  = true;
      }

      // lights
      setDot(txDot, !!status.tx);
      setDot(rxDot, !!status.rx);
      runPill.textContent = status.tx ? 'Running (TX)' : (status.rx ? 'Running (RX)' : 'Running');
      setRunPillState(!!status.running, !!status.rx, !!status.tx);

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
    })
    .catch(()=>{ uiRunning = false; });
}, 500);

// ====== Smooth meters (fast) ======
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
  document.getElementById('level-text').textContent = shownRxDb.toFixed(1) + ' dB';

  requestAnimationFrame(meterTick);
}
requestAnimationFrame(meterTick);

setInterval(() => {
  if (!uiRunning) { targetRxDb = -60; targetTxDb = -60; return; }

  fetch(apiUrl('meter'), { cache: "no-store" })
    .then(r => r.json())
    .then(m => {
      if (!m.running) { targetRxDb = -60; targetTxDb = -60; return; }

      targetRxDb = (typeof m.rx_db === 'number') ? m.rx_db : -60;
      targetTxDb = (typeof m.tx_db === 'number') ? m.tx_db : -60;

      const clip = document.getElementById('clip-alert');
      const limit = document.getElementById('limit-alert');
      if (m.clipping) clip.classList.remove('hidden'); else clip.classList.add('hidden');
      if (m.limiting) limit.classList.remove('hidden'); else limit.classList.add('hidden');
    })
    .catch(()=>{});
}, 50);

// ====== Conservative Config Lock + Live Updates + Save ======
const lockEl = document.getElementById("cfg-lock");
const lockState = document.getElementById("lock-state");
const saveBtn = document.getElementById("save-btn");
const saveStatus = document.getElementById("save-status");
const restartBadge = document.getElementById("restart-badge");

// things we need for conditional disables
const dualOutputEl = document.getElementById("dual-output");
const out2El = document.getElementById("output-device-2");
const courtesyToneEl = document.getElementById("courtesy-tone");
const courtesyVolEl = document.getElementById("courtesy-vol");
const cwEnabledEl = document.getElementById("cw-enabled");
const totLockoutEl = document.getElementById("tot-lockout");
const lockoutTimeEl = document.getElementById("lockout-time");
const pttModeEl = document.getElementById("ptt-mode");
const pttDevEl = document.getElementById("ptt-device");
const pttPinEl = document.getElementById("ptt-pin");

// helper: POST JSON
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

// helper: tiny pill feedback
function setSavePill(text, cls){
  saveStatus.style.display = "inline-flex";
  saveStatus.textContent = text;
  saveStatus.classList.remove("ok","warn","bad");
  if (cls) saveStatus.classList.add(cls);
  setTimeout(() => { saveStatus.style.display = "none"; }, 1500);
}

// keep dependent fields disabled when feature is off (and also when locked)
function syncConditionalDisables(){
  const locked = lockEl.checked;

  // HP cutoff should be disabled if HPF off or locked
  const hpToggle = document.getElementById("highpass");
  const hpCutoff = document.getElementById("highpass-cutoff");
  if (hpToggle && hpCutoff){
    hpCutoff.disabled = locked || !hpToggle.checked;
  }

  // dual output device picker
  if (out2El && dualOutputEl){
    out2El.disabled = locked || !dualOutputEl.checked;
  }

  // courtesy volume only when courtesy tone enabled
  if (courtesyVolEl && courtesyToneEl){
    courtesyVolEl.disabled = locked || !courtesyToneEl.checked;
  }

  // CW fields only when CW enabled
  ["cw-speed","cw-pitch","cw-vol"].forEach(id => {
    const el = document.getElementById(id);
    if (el && cwEnabledEl){
      el.disabled = locked || !cwEnabledEl.checked;
    }
  });

  // TOT lockout time only when lockout enabled
  if (lockoutTimeEl && totLockoutEl){
    lockoutTimeEl.disabled = locked || !totLockoutEl.checked;
  }

  // PTT fields: conservative UX (VOX doesn't need device/pin)
  if (pttModeEl && pttDevEl && pttPinEl){
    const mode = String(pttModeEl.value || "").toUpperCase();
    const pttFieldsShouldBeEnabled = !locked && (mode === "CM108");
    pttDevEl.disabled = !pttFieldsShouldBeEnabled;
    pttPinEl.disabled = !pttFieldsShouldBeEnabled;
  }
}

function setLockedUI(locked){
  lockState.textContent = locked ? "LOCKED" : "UNLOCKED";
  lockState.style.color = locked ? "#ffcc33" : "#3dff47";

  const form = document.getElementById('config-form');
  const configCard = form.querySelector('.grid .card:nth-child(2)');

  if (configCard){
    configCard.querySelectorAll('input, select, textarea, button').forEach(el => {
      // IMPORTANT: never disable the lock checkbox itself
      if (el === lockEl) return;

      // allow Save and tab controls always
      if (el.id === "save-btn") return;
      if (el.classList.contains("tab-btn")) return;

      // disable everything else in config card when locked
      el.disabled = locked;
    });
  }

  // Optional: lock device selects too
  const inputDev = document.getElementById('input-device');
  const outputDev = document.getElementById('output-device');
  if (inputDev) inputDev.disabled = locked;
  if (outputDev) outputDev.disabled = locked;

  // after base enable/disable, re-apply dependent logic
  syncConditionalDisables();
}

// initial state
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

// one change handler (with confirm on unlock)
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

// debounced live updates
const debounceTimers = new Map();

function sendLiveUpdate(el){
  if (lockEl.checked) return;
  const key = el.dataset.key;
  if (!key) return;

  const value = (el.type === "checkbox") ? el.checked : el.value;

  postJSON(apiUrl("config/live"), {key, value})
    .then(resp => {
      if (resp && resp.status === "restart_required"){
        restartBadge.style.display = "inline-flex";
        el.classList.add("needs-restart");
      }
    })
    .catch(()=>{});
}

function scheduleUpdate(el){
  const key = el.dataset.key || el.id;
  if (debounceTimers.has(key)) clearTimeout(debounceTimers.get(key));
  debounceTimers.set(key, setTimeout(() => sendLiveUpdate(el), 120));
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

// Save button
saveBtn.addEventListener("click", () => {
  fetch(apiUrl("config/apply"), {method:"POST"})
    .then(r => r.json())
    .then(d => {
      if (d && d.status === "ok") setSavePill("Saved ✅", "ok");
      else setSavePill("Save failed ❌", "bad");
    })
    .catch(()=> setSavePill("Save failed ❌", "bad"));
});

// ensure dependent disables are correct on load
syncConditionalDisables();

async function updateSystemStats() {
  try {
    const res = await fetch(apiUrl("api/stats"));
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

let lastLogId = 0;

async function fetchLogs() {
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

setInterval(fetchLogs, 1000);
fetchLogs();

async function clearLogs() {
  await fetch(apiUrl("logs/clear"), { method: "POST" });

  const container = document.getElementById("logContainer");
  if (container) container.innerHTML = "";

  lastLogId = 0;
}

async function updateMaintenanceStatus() {
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

setInterval(updateMaintenanceStatus, 2000);
updateMaintenanceStatus();
