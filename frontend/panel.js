// panel.js — Recibe estado via WebSocket y actualiza la UI

const ws = new WebSocket(`ws://${location.host}/ws`);
let currentPlayer = null;

ws.onopen = () => setStatus("Conectado", "#68d391");
ws.onclose = () => setStatus("Desconectado", "#fc8181");
ws.onerror = () => setStatus("Error de conexion", "#fc8181");

ws.onmessage = (event) => {
  const state = JSON.parse(event.data);
  updateUI(state);
};

function updateUI(state) {
  // Conteo
  const c = state.counting || {};
  setText("running-count", c.running_count ?? 0);
  setText("true-count", (c.true_count ?? 0).toFixed(1));
  setText("decks-remaining", (c.decks_remaining ?? 0).toFixed(1));
  setText("cards-seen", c.cards_seen ?? 0);

  // Jugadores
  const container = document.getElementById("players-container");
  container.innerHTML = "";
  currentPlayer = state.current_player;

  // Dealer
  const dealer = state.dealer || {};
  container.appendChild(makePlayerBlock("Dealer", dealer.cards || [], dealer.total, false, false));

  // Jugadores
  for (const [pid, hand] of Object.entries(state.players || {})) {
    const isActive = pid === currentPlayer;
    const block = makePlayerBlock(pid, hand.cards || [], hand.total, hand.bust, hand.blackjack, isActive);
    container.appendChild(block);

    // Sugerencia del jugador activo
    if (isActive && hand.suggestion) {
      updateSuggestion(hand.suggestion);
    }
  }

  // Estado de juego
  setStatus(`Estado: ${state.state ?? "—"}`, "#a0aec0");
}

function makePlayerBlock(name, cards, total, bust, bj, active = false) {
  const div = document.createElement("div");
  div.className = "player-block";
  if (active) div.style.borderLeft = "3px solid #63b3ed";

  const nameEl = document.createElement("div");
  nameEl.className = "player-name";
  nameEl.textContent = name.replace("_", " ").toUpperCase() + (active ? " ◀" : "");

  const cardsEl = document.createElement("div");
  cardsEl.className = "player-cards";
  cardsEl.textContent = cards.length ? cards.join("  ") : "—";

  const totalEl = document.createElement("div");
  totalEl.className = "player-total" + (bust ? " bust" : bj ? " bj" : "");
  totalEl.textContent = bust ? `${total} BUST` : bj ? "BLACKJACK!" : `Total: ${total}`;

  div.appendChild(nameEl);
  div.appendChild(cardsEl);
  div.appendChild(totalEl);
  return div;
}

function updateSuggestion(action) {
  const box = document.getElementById("suggestion-box");
  const text = document.getElementById("suggestion-action");
  text.textContent = action;
  box.className = "suggestion-box " + action.toLowerCase();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setStatus(msg, color = "#a0aec0") {
  const bar = document.getElementById("status-bar");
  if (bar) { bar.textContent = msg; bar.style.color = color; }
}

async function sendStand() {
  if (!currentPlayer) return;
  await fetch("/api/stand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_id: currentPlayer }),
  });
}

async function sendReset() {
  await fetch("/api/reset", { method: "POST" });
}
