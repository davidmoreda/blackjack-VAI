// panel.js — UI de Blackjack VAI

const ws = new WebSocket(`ws://${location.host}/ws`);
let currentPlayer = null;
let gameActive = false;
let lastState = null;

// Setup config (selecciones del usuario, antes de aplicar)
const PLAYERS_OPTIONS = [1, 2, 3, 4];
const DECKS_OPTIONS   = [1, 2, 4, 6, 8];
const SYSTEMS_OPTIONS = [
  { id: "hilo",   label: "Hi-Lo" },
  { id: "ko",     label: "KO" },
  { id: "omega2", label: "Omega II" },
];
let appliedConfig = { num_players: 1, num_decks: 6, counting_system: "hilo" };
let pendingConfig = { ...appliedConfig };

// ── Setup panel: render botones y manejo de seleccion ─────────────────
function renderSetupPanel() {
  renderOptGroup("opt-players", PLAYERS_OPTIONS.map(n => ({ id: n, label: String(n) })),
                 "num_players");
  renderOptGroup("opt-decks", DECKS_OPTIONS.map(n => ({ id: n, label: String(n) })),
                 "num_decks");
  renderOptGroup("opt-system", SYSTEMS_OPTIONS, "counting_system", "system");
  refreshOptHighlights();
  refreshApplyButton();
}

function renderOptGroup(containerId, options, key, extraClass = "") {
  const c = document.getElementById(containerId);
  if (!c) return;
  c.innerHTML = "";
  options.forEach(opt => {
    const btn = document.createElement("button");
    btn.className = "opt-btn" + (extraClass ? " " + extraClass : "");
    btn.textContent = opt.label;
    btn.dataset.key = key;
    btn.dataset.value = opt.id;
    btn.onclick = () => {
      pendingConfig[key] = opt.id;
      refreshOptHighlights();
      refreshApplyButton();
    };
    c.appendChild(btn);
  });
}

function refreshOptHighlights() {
  document.querySelectorAll("#setup-panel .opt-btn").forEach(btn => {
    const k = btn.dataset.key;
    const v = btn.dataset.value;
    const cur = String(pendingConfig[k]);
    if (String(v) === cur) btn.classList.add("selected");
    else                   btn.classList.remove("selected");
  });
}

function refreshApplyButton() {
  const btn = document.getElementById("btn-apply-config");
  if (!btn) return;
  const dirty =
    pendingConfig.num_players     !== appliedConfig.num_players  ||
    pendingConfig.num_decks       !== appliedConfig.num_decks    ||
    pendingConfig.counting_system !== appliedConfig.counting_system;
  if (dirty) {
    btn.classList.add("dirty");
    btn.textContent = "Aplicar y reiniciar partida";
    btn.disabled = false;
  } else {
    btn.classList.remove("dirty");
    btn.textContent = "Configuración aplicada";
    btn.disabled = true;
  }
}

async function applyConfig() {
  const r = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pendingConfig),
  });
  if (!r.ok) {
    alert("No se pudo aplicar la configuración");
    return;
  }
  const data = await r.json();
  appliedConfig = { ...data.config };
  pendingConfig = { ...appliedConfig };
  refreshApplyButton();
}

async function loadCurrentConfig() {
  try {
    const r = await fetch("/api/config");
    if (!r.ok) return;
    const c = await r.json();
    appliedConfig = { ...c };
    pendingConfig = { ...c };
    refreshOptHighlights();
    refreshApplyButton();
  } catch (e) {
    /* primera carga, sin red, ignorar */
  }
}

// Inicializar setup panel al cargar
document.addEventListener("DOMContentLoaded", () => {
  renderSetupPanel();
  loadCurrentConfig();
});

ws.onopen  = () => setWsStatus("EN VIVO", true);
ws.onclose = () => setWsStatus("DESCONECTADO", false);
ws.onerror = () => setWsStatus("ERROR", false);
ws.onmessage = (e) => updateUI(JSON.parse(e.data));

function setWsStatus(msg, live) {
  const dot = document.getElementById("status-dot");
  const lbl = document.getElementById("ws-status");
  if (dot) dot.className = "status-dot" + (live ? " live" : "");
  if (lbl) lbl.textContent = msg;
}

function updateUI(state) {
  lastState = state;
  gameActive = !!state.game_active;

  // Badge + boton inicio
  const badge    = document.getElementById("game-badge");
  const btnStart = document.getElementById("btn-start");
  if (gameActive) {
    badge.textContent = "ACTIVO";
    badge.className   = "active";
    btnStart.textContent = "♦ REINICIAR PARTIDA";
    btnStart.className   = "btn btn-start active";
  } else {
    badge.textContent = "EN ESPERA";
    badge.className   = "";
    btnStart.textContent = "♠ EMPEZAR PARTIDA";
    btnStart.className   = "btn btn-start";
  }

  document.getElementById("btn-stand").disabled = !gameActive || !currentPlayer;
  document.getElementById("btn-reset").disabled = !gameActive;
  document.getElementById("btn-undo").disabled  = !state.history?.length;

  // Marcador
  updateScoreboard(state.score);

  // Conteo
  const c    = state.counting || {};
  const deck = state.deck     || {};
  const rc = c.running_count ?? 0;
  const tc = c.true_count    ?? 0;

  const countTitle = document.getElementById("count-title");
  if (countTitle) {
    const sys = (state.config?.counting_system || "").toUpperCase() || "";
    countTitle.textContent = sys ? `Conteo ${sys}` : "Conteo";
  }

  const rcEl = document.getElementById("running-count");
  if (rcEl) {
    rcEl.textContent = rc > 0 ? `+${rc}` : rc;
    rcEl.className   = rc > 0 ? "positive" : rc < 0 ? "negative" : "";
  }
  setText("true-count",      tc >= 0 ? `+${tc.toFixed(1)}` : tc.toFixed(1));
  setText("decks-remaining", (c.decks_remaining ?? 0).toFixed(1));
  setText("cards-seen",      `${c.cards_seen ?? 0}` + (deck.total_remaining ? ` / ${deck.total_remaining + (c.cards_seen||0)}` : ""));

  // Barra de ventaja
  const fill = document.getElementById("adv-bar-fill");
  if (fill) {
    const pct = Math.min(Math.max((tc + 6) / 12, 0), 1) * 100;
    fill.style.width = pct + "%";
    fill.style.background =
      tc >= 2  ? "#4ade80" :
      tc <= -2 ? "#f87171" : "#94a3b8";
  }

  // Mesa
  const container = document.getElementById("players-container");
  container.innerHTML = "";
  currentPlayer = state.current_player;

  const dealer = state.dealer || {};
  container.appendChild(makePlayerBlock(
    "Dealer", dealer.cards || [], dealer.total, false, false, false
  ));

  for (const [pid, hand] of Object.entries(state.players || {})) {
    const isActive = pid === currentPlayer;
    container.appendChild(makePlayerBlock(
      pid, hand.cards || [], hand.total,
      hand.bust, hand.blackjack, isActive,
      hand.suggestion, hand.evs
    ));
  }

  // Sugerencia principal — siempre del primer jugador (j1) o del jugador activo
  const focusPid = currentPlayer || Object.keys(state.players || {})[0];
  const focus = focusPid ? state.players?.[focusPid] : null;
  if (focus) {
    updateSuggestion(focus);
  } else {
    updateSuggestion(null);
  }

  updateAlerts(state);
  updateHistory(state.history || []);
}

function updateScoreboard(score) {
  if (!score) return;
  setText("score-dealer", score.dealer);

  // Si hay un solo jugador mostramos "Tu" (ya esta en HTML).
  // Si hay varios, pintamos celdas dinamicamente.
  const players = score.players || {};
  const playerKeys = Object.keys(players);
  const board = document.getElementById("scoreboard");

  if (playerKeys.length === 1) {
    // Modo simple: solo Dealer / Tu / Push
    setText("score-player", players[playerKeys[0]]);
    setText("score-push", sumValues(score.pushes));
  } else {
    // Multi-jugador: regenerar layout
    board.innerHTML = "";
    board.appendChild(makeScoreCell("dealer", "Dealer", score.dealer));
    for (const pid of playerKeys) {
      board.appendChild(makeScoreCell("player", pid.replace("player_", "P"), players[pid]));
    }
    const totalPush = sumValues(score.pushes);
    if (totalPush > 0) {
      board.appendChild(makeScoreCell("push", "Push", totalPush));
    }
  }

  setText("rounds-count", `${score.rounds || 0} rondas`);
}

function makeScoreCell(cls, label, val) {
  const cell = document.createElement("div");
  cell.className = `score-cell ${cls}`;
  cell.innerHTML = `
    <div class="score-cell-label">${label}</div>
    <div class="score-cell-val">${val}</div>
  `;
  return cell;
}

function sumValues(obj) {
  return Object.values(obj || {}).reduce((a, b) => a + b, 0);
}

function makePlayerBlock(name, cards, total, bust, bj, active, suggestion, evs) {
  const wrap = document.createElement("div");
  wrap.className = "player-block" + (active ? " active-player" : "");

  const header = document.createElement("div");
  header.className = "player-header";

  const nameEl = document.createElement("div");
  nameEl.className = "player-name";
  nameEl.textContent = name.replace("_", " ").toUpperCase() + (active ? "  ◀" : "");

  const totalEl = document.createElement("div");
  totalEl.className = "player-total-badge" + (bust ? " bust" : bj ? " bj" : "");
  totalEl.textContent = bust ? `${total} BUST` : bj ? "BJ!" : total > 0 ? total : "—";

  header.appendChild(nameEl);
  header.appendChild(totalEl);

  const cardsRow = document.createElement("div");
  cardsRow.className = "player-cards-row";
  if (cards.length) {
    cards.forEach(c => {
      const tag = document.createElement("span");
      const isRed = c.includes("Hearts") || c.includes("Diamonds");
      tag.className = "card-tag" + (isRed ? " red" : "");
      tag.textContent = formatCard(c);
      cardsRow.appendChild(tag);
    });
  } else {
    cardsRow.textContent = "—";
  }

  wrap.appendChild(header);
  wrap.appendChild(cardsRow);

  if (suggestion) {
    const sugEl = document.createElement("div");
    const colors = { HIT:"#f87171", STAND:"#4ade80", DOUBLE:"#60a5fa", SPLIT:"#a78bfa", WAIT:"#94a3b8" };
    sugEl.style.cssText = `margin-top:6px;font-size:0.7rem;font-weight:700;letter-spacing:2px;color:${colors[suggestion]||"#94a3b8"}`;
    let txt = "→ " + suggestion;
    if (evs && evs[suggestion] !== undefined) {
      txt += `  (${formatEv(evs[suggestion])})`;
    }
    sugEl.textContent = txt;
    wrap.appendChild(sugEl);
  }

  return wrap;
}

function formatCard(label) {
  const suits = { Spades: "♠", Hearts: "♥", Diamonds: "♦", Clubs: "♣" };
  const parts = label.split(" ");
  if (parts.length >= 2) {
    const suit = suits[parts[parts.length - 1]] || "";
    return parts[0] + suit;
  }
  return label;
}

function formatEv(v) {
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

function updateSuggestion(pdata) {
  const box  = document.getElementById("suggestion-box");
  const text = document.getElementById("suggestion-action");
  const evsEl = document.getElementById("suggestion-ev");
  if (!box || !text) return;

  if (!pdata) {
    text.textContent = "—";
    box.className = "wait";
    if (evsEl) evsEl.innerHTML = "";
    return;
  }

  // Estado especial primero
  if (pdata.bust)        { text.textContent = `BUST ${pdata.total}`;  box.className = "wait";  if (evsEl) evsEl.innerHTML = ""; return; }
  if (pdata.blackjack)   { text.textContent = "BLACKJACK!";           box.className = "double"; if (evsEl) evsEl.innerHTML = ""; return; }
  if (pdata.stood)       { text.textContent = `STAND ${pdata.total}`; box.className = "stand";  if (evsEl) evsEl.innerHTML = ""; return; }

  const sug = pdata.suggestion;
  if (!sug) {
    text.textContent = (pdata.cards?.length ? "ESPERANDO" : "SIN CARTAS");
    box.className = "wait";
    if (evsEl) evsEl.innerHTML = "";
    return;
  }

  text.textContent = sug;
  box.className    = "wait";
  const cls = sug.toLowerCase();
  if (["hit","stand","double","split"].includes(cls)) box.classList.add(cls);

  // EV detallado debajo
  if (evsEl && pdata.evs) {
    const parts = [];
    for (const [act, val] of Object.entries(pdata.evs)) {
      const cls = val > 0 ? "ev-positive" : val < 0 ? "ev-negative" : "";
      const isBest = act === sug;
      parts.push(`<span class="${cls}${isBest ? ' ev-best' : ''}">${act} ${formatEv(val)}</span>`);
    }
    evsEl.innerHTML = parts.join("  ·  ");
  } else if (evsEl) {
    evsEl.innerHTML = "";
  }
}

function updateAlerts(state) {
  const container = document.getElementById("alerts-container");
  if (!container) return;
  container.innerHTML = "";

  const checks = [];
  const dealer = state.dealer || {};
  if (dealer.total > 21)        checks.push({ cls: "bust",      msg: `DEALER BUST — ${dealer.total}` });
  else if (dealer.total === 21) checks.push({ cls: "twentyone", msg: "DEALER — 21 !" });

  for (const [pid, hand] of Object.entries(state.players || {})) {
    const label = pid.replace("_", " ").toUpperCase();
    if (hand.bust)              checks.push({ cls: "bust",      msg: `${label} BUST — ${hand.total}` });
    else if (hand.blackjack)    checks.push({ cls: "blackjack", msg: `${label} — BLACKJACK !` });
    else if (hand.total === 21) checks.push({ cls: "twentyone", msg: `${label} — 21 !` });
  }

  checks.forEach(({ cls, msg }) => {
    const div = document.createElement("div");
    div.className = `hand-alert ${cls}`;
    div.textContent = msg;
    container.appendChild(div);
  });
}

function updateHistory(history) {
  const container = document.getElementById("history-list");
  if (!container) return;
  if (!history.length) {
    container.innerHTML = '<span class="seen-empty">Sin detecciones</span>';
    return;
  }
  container.innerHTML = "";
  history.forEach((entry, i) => {
    const row = document.createElement("div");
    row.className = "history-row";

    const idx = document.createElement("div");
    idx.className = "history-idx";
    idx.textContent = `${i + 1}`;

    const zone = document.createElement("div");
    zone.className = `history-zone ${entry.zone}`;
    zone.textContent = entry.zone === "dealer" ? "D" : entry.zone.replace("player_", "P");

    const card = document.createElement("div");
    card.className = "history-card";
    card.textContent = formatCard(entry.card);

    const del = document.createElement("button");
    del.className = "history-del";
    del.textContent = "×";
    del.title = "Eliminar esta carta";
    del.onclick = () => sendRemove(i);

    row.appendChild(idx);
    row.appendChild(zone);
    row.appendChild(card);
    row.appendChild(del);
    container.appendChild(row);
  });
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

async function sendStart() {
  await fetch("/api/start", { method: "POST" });
}

async function sendStand() {
  if (!currentPlayer || !gameActive) return;
  await fetch("/api/stand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_id: currentPlayer }),
  });
}

async function sendReset() {
  if (!gameActive) return;
  await fetch("/api/reset", { method: "POST" });
}

async function sendUndo() {
  await fetch("/api/undo", { method: "POST" });
}

async function sendRemove(idx) {
  await fetch(`/api/remove/${idx}`, { method: "POST" });
}
