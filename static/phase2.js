// Phase 2 Living Hero Frontend Client
let ws;
let currentState = null;

const canvas = document.getElementById("village-canvas");
const ctx = canvas.getContext("2d");

// Connect WebSocket
function connectWS() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/phase2`);

  ws.onopen = () => {
    document.getElementById("connection-status").textContent = "Connected";
    document.getElementById("connection-status").style.color = "#10b981";
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "phase2_update") {
      currentState = data;
      updateUI(data);
      render(data);
    }
  };

  ws.onclose = () => {
    document.getElementById("connection-status").textContent = "Reconnecting...";
    document.getElementById("connection-status").style.color = "#ef4444";
    setTimeout(connectWS, 1500);
  };
}

function updateUI(data) {
  // Status Bar
  document.getElementById("stat-tick").textContent = data.tick;
  document.getElementById("stat-time").textContent = data.time_of_day;
  document.getElementById("stat-action").textContent = data.action_name;

  // Active Goal Badge
  const badge = document.getElementById("goal-badge");
  const goal = data.goal;
  if (goal === "HYDRATE") {
    badge.textContent = "💧 HYDRATE (Seeking Water)";
    badge.style.background = "rgba(6, 182, 212, 0.2)";
    badge.style.color = "#06b6d4";
    badge.style.borderColor = "#06b6d4";
  } else if (goal === "RETURN_HOME") {
    badge.textContent = "⛺ RETURNING HOME (Deposit/Rest)";
    badge.style.background = "rgba(250, 204, 21, 0.2)";
    badge.style.color = "#facc15";
    badge.style.borderColor = "#facc15";
  } else if (goal === "REST") {
    badge.textContent = "💤 RESTING IN SHELTER";
    badge.style.background = "rgba(56, 189, 248, 0.2)";
    badge.style.color = "#38bdf8";
    badge.style.borderColor = "#38bdf8";
  } else if (goal === "SURVIVE") {
    badge.textContent = "🚨 SURVIVE (Critical Vitals!)";
    badge.style.background = "rgba(239, 68, 68, 0.2)";
    badge.style.color = "#ef4444";
    badge.style.borderColor = "#ef4444";
  } else {
    badge.textContent = "🍏 FORAGING & HARVESTING";
    badge.style.background = "rgba(16, 185, 129, 0.2)";
    badge.style.color = "#10b981";
    badge.style.borderColor = "#10b981";
  }

  // Vitals
  const v = data.vitals;
  document.getElementById("val-health").textContent = `${Math.round(v.health)}%`;
  document.getElementById("bar-health").style.width = `${Math.max(0, Math.min(100, v.health))}%`;
  document.getElementById("bar-health").style.background = v.health < 35 ? "#ef4444" : "#10b981";

  document.getElementById("val-thirst").textContent = `${Math.round(v.thirst)}%`;
  document.getElementById("bar-thirst").style.width = `${Math.max(0, Math.min(100, v.thirst))}%`;
  document.getElementById("bar-thirst").style.background = v.thirst >= 70 ? "#ef4444" : "#06b6d4";

  document.getElementById("val-hunger").textContent = `${Math.round(v.hunger)}%`;
  document.getElementById("bar-hunger").style.width = `${Math.max(0, Math.min(100, v.hunger))}%`;
  document.getElementById("bar-hunger").style.background = v.hunger >= 70 ? "#ef4444" : "#facc15";

  document.getElementById("val-energy").textContent = `${Math.round(v.energy)}%`;
  document.getElementById("bar-energy").style.width = `${Math.max(0, Math.min(100, v.energy))}%`;
  document.getElementById("bar-energy").style.background = v.energy <= 25 ? "#ef4444" : "#10b981";

  // Backpack Slots
  const bp = data.backpack;
  const cap = data.backpack_capacity || 5;
  const total = bp.food + bp.wood;
  document.getElementById("backpack-capacity-text").textContent = `${total} / ${cap}`;

  const bpContainer = document.getElementById("backpack-container");
  let slotsHtml = "";
  let foodLeft = bp.food;
  let woodLeft = bp.wood;

  for (let i = 0; i < cap; i++) {
    if (foodLeft > 0) {
      slotsHtml += `<div class="backpack-slot filled">🍏</div>`;
      foodLeft--;
    } else if (woodLeft > 0) {
      slotsHtml += `<div class="backpack-slot filled">🪵</div>`;
      woodLeft--;
    } else {
      slotsHtml += `<div class="backpack-slot">░</div>`;
    }
  }
  bpContainer.innerHTML = slotsHtml;

  // Village Stockpile
  document.getElementById("stockpile-food").textContent = data.stockpile.food;
  document.getElementById("stockpile-wood").textContent = data.stockpile.wood;

  // Events Feed
  if (data.events && data.events.length > 0) {
    const logEl = document.getElementById("event-log-container");
    data.events.forEach((ev) => {
      const item = document.createElement("div");
      item.className = "event-item";
      item.textContent = `[T${data.tick}] ${ev}`;
      logEl.insertBefore(item, logEl.firstChild);
    });
    while (logEl.children.length > 25) {
      logEl.removeChild(logEl.lastChild);
    }
  }

  // Play/Pause button text
  document.getElementById("btn-play-pause").textContent = data.is_paused ? "▶ Play" : "⏸ Pause";
}

function render(data) {
  const size = data.grid_size || 14;
  const w = canvas.width;
  const h = canvas.height;
  const cell = w / size;

  ctx.clearRect(0, 0, w, h);

  // 1. Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell, h);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * cell);
    ctx.lineTo(w, i * cell);
    ctx.stroke();
  }

  // 2. Home Base Zone
  const [hr, hc] = data.home_pos;
  ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
  ctx.fillRect((hc - 1) * cell, (hr - 1) * cell, cell * 3, cell * 3);
  ctx.strokeStyle = "rgba(245, 158, 11, 0.3)";
  ctx.strokeRect((hc - 1) * cell, (hr - 1) * cell, cell * 3, cell * 3);

  // Draw Home Base Icon
  drawEmoji("⛺", hc, hr, cell, 0.8);

  // 3. Water Spring / Lake
  data.water_tiles.forEach(([wr, wc]) => {
    ctx.fillStyle = "rgba(6, 182, 212, 0.35)";
    ctx.fillRect(wc * cell, wr * cell, cell, cell);
    drawEmoji("💧", wc, wr, cell, 0.65);
  });

  // 4. Hazards
  data.hazards.forEach(([hzr, hzc]) => {
    ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
    ctx.fillRect(hzc * cell, hzr * cell, cell, cell);
    drawEmoji("⚠️", hzc, hzr, cell, 0.65);
  });

  // 5. Timber Trees
  data.trees.forEach(([tr, tc]) => {
    drawEmoji("🌲", tc, tr, cell, 0.75);
  });

  // 6. Food Bushes
  data.foods.forEach(([fr, fc]) => {
    drawEmoji("🍏", fc, fr, cell, 0.7);
  });

  // 7. Hero Agent
  const [ar, ac] = data.agent_pos;
  // Hero Aura
  ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
  ctx.beginPath();
  ctx.arc((ac + 0.5) * cell, (ar + 0.5) * cell, cell * 0.45, 0, Math.PI * 2);
  ctx.fill();

  const heroEmoji = (data.goal === "REST" && ar === hr && ac === hc) ? "😴" : "🧙";
  drawEmoji(heroEmoji, ac, ar, cell, 0.85);
}

function drawEmoji(emoji, c, r, cell, scale = 0.7) {
  ctx.font = `${Math.floor(cell * scale)}px -apple-system, BlinkMacSystemFont, "Segoe UI Emoji", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(emoji, (c + 0.5) * cell, (r + 0.5) * cell);
}

// Controls Setup
document.getElementById("btn-play-pause").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "toggle_pause" }));
  }
});

document.getElementById("btn-step").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "step_once" }));
  }
});

document.getElementById("btn-reset").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "reset" }));
  }
});

const speedSlider = document.getElementById("speed-slider");
const speedLabel = document.getElementById("speed-label");
speedSlider.addEventListener("input", (e) => {
  const val = e.target.value;
  speedLabel.textContent = `${val}x`;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "set_speed", speed: parseFloat(val) }));
  }
});

// Start WebSocket on Load
window.addEventListener("DOMContentLoaded", connectWS);
