// panel.js

const ws = new WebSocket(`ws://${location.host}/ws`);
let currentPlayer = null;
let gameActive = false;

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
  gameActive = !!state.game_active;

  // Badge + botón inicio
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
    btnStart.textContent = "♠ EMPEZAR A JUGAR";
    btnStart.className   = "btn btn-start";
  }

  document.getElementById("btn-stand").disabled = !gameActive;
  document.getElementById("btn-reset").disabled = !gameActive;

  // Conteo
  const c  = state.counting || {};
  const rc = c.running_count ?? 0;
  const tc = c.true_count    ?? 0;

  const rcEl = document.getElementById("running-count");
  if (rcEl) {
    rcEl.textContent = rc > 0 ? `+${rc}` : rc;
    rcEl.className   = rc > 0 ? "positive" : rc < 0 ? "negative" : "";
  }
  setText("true-count",      tc >= 0 ? `+${tc.toFixed(1)}` : tc.toFixed(1));
  setText("decks-remaining", (c.decks_remaining ?? 0).toFixed(1));
  setText("cards-seen",      c.cards_seen ?? 0);

  // Barra de ventaja: true count [-6, +6] → 0–100%
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
  container.appendChild(makePlayerBlock("Dealer", dealer.cards || [], dealer.total, false, false, false));

  for (const [pid, hand] of Object.entries(state.players || {})) {
    const isActive = pid === currentPlayer;
    container.appendChild(makePlayerBlock(pid, hand.cards || [], hand.total, hand.bust, hand.blackjack, isActive, hand.suggestion));
  }

  // Sugerencia del jugador activo en el box grande
  const activeSug = state.players?.[currentPlayer]?.suggestion;
  if (gameActive && activeSug) updateSuggestion(activeSug);
  else if (!gameActive) updateSuggestion("—", "wait");

  updateAlerts(state);
  updateSeenCards(state.seen_cards || []);
}

function makePlayerBlock(name, cards, total, bust, bj, active, suggestion) {
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

  if (suggestion && suggestion !== "WAIT") {
    const sugEl = document.createElement("div");
    const colors = { HIT:"#f87171", STAND:"#4ade80", DOUBLE:"#60a5fa", SPLIT:"#a78bfa" };
    sugEl.style.cssText = `margin-top:5px;font-size:0.7rem;font-weight:700;letter-spacing:2px;color:${colors[suggestion]||"#94a3b8"}`;
    sugEl.textContent = "→ " + suggestion;
    wrap.appendChild(sugEl);
  }

  return wrap;
}

function formatCard(label) {
  // "A Spades" → "A♠",  "10 Hearts" → "10♥"
  const suits = { Spades: "♠", Hearts: "♥", Diamonds: "♦", Clubs: "♣" };
  const parts = label.split(" ");
  if (parts.length >= 2) {
    const suit = suits[parts[parts.length - 1]] || "";
    return parts[0] + suit;
  }
  return label;
}

function updateSuggestion(action, forceClass) {
  const box  = document.getElementById("suggestion-box");
  const text = document.getElementById("suggestion-action");
  if (!box || !text) return;
  text.textContent = action;
  box.className    = "wait";
  const cls = forceClass || action.toLowerCase();
  if (["hit","stand","double","split","wait"].includes(cls)) box.classList.add(cls);
}

function updateAlerts(state) {
  const container = document.getElementById("alerts-container");
  if (!container) return;
  container.innerHTML = "";

  const checks = [];

  // Dealer
  const dealer = state.dealer || {};
  if (dealer.total > 21)       checks.push({ name: "DEALER", cls: "bust",      msg: `DEALER BUST — ${dealer.total}` });
  else if (dealer.total === 21) checks.push({ name: "DEALER", cls: "twentyone", msg: "DEALER — 21 !" });

  // Jugadores
  for (const [pid, hand] of Object.entries(state.players || {})) {
    const label = pid.replace("_", " ").toUpperCase();
    if (hand.bust)              checks.push({ name: pid, cls: "bust",      msg: `${label} BUST — ${hand.total}` });
    else if (hand.blackjack)    checks.push({ name: pid, cls: "blackjack", msg: `${label} — BLACKJACK !` });
    else if (hand.total === 21) checks.push({ name: pid, cls: "twentyone", msg: `${label} — 21 !` });
  }

  checks.forEach(({ cls, msg }) => {
    const div = document.createElement("div");
    div.className = `hand-alert ${cls}`;
    div.textContent = msg;
    container.appendChild(div);
  });
}

function updateSeenCards(cards) {
  const container = document.getElementById("seen-cards-container");
  if (!container) return;
  if (!cards.length) {
    container.innerHTML = '<span class="seen-empty">Sin detecciones</span>';
    return;
  }
  const grid = document.createElement("div");
  grid.className = "seen-grid";
  cards.forEach(c => {
    const chip = document.createElement("span");
    const isRed = c.includes("Hearts") || c.includes("Diamonds");
    chip.className = "seen-chip" + (isRed ? " red" : "");
    chip.textContent = formatCard(c);
    grid.appendChild(chip);
  });
  container.innerHTML = "";
  container.appendChild(grid);
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
