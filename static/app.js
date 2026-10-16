// Hybrid ERL Frontend Client
let ws;
let currentState = null;
let currentView = "split"; // 'world', 'neural', 'split'

// Canvases & Contexts
const worldCanvas = document.getElementById("world-canvas");
const worldCtx = worldCanvas.getContext("2d");
const brainCanvas = document.getElementById("brain-canvas");
const brainCtx = brainCanvas.getContext("2d");

// Connect WebSocket
function connectWS() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    document.getElementById("connection-status").textContent = "Connected";
    document.getElementById("connection-status").style.color = "#10b981";
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "state_update") {
      currentState = data;
      updateUI(data);
      render();
    }
  };

  ws.onclose = () => {
    document.getElementById("connection-status").textContent = "Reconnecting...";
    document.getElementById("connection-status").style.color = "#ef4444";
    setTimeout(connectWS, 2000);
  };
}

// Update UI Telemetry & Leaderboard
function updateUI(data) {
  document.getElementById("stat-gen").textContent = data.gen;
  document.getElementById("stat-tick").textContent = `${data.tick} / ${data.max_ticks}`;
  document.getElementById("stat-top-food").textContent = data.stats.top_food;
  document.getElementById("stat-avg-food").textContent = data.stats.avg_food;
  document.getElementById("stat-lineage").textContent = data.stats.top_lineage;

  if (data.brain) {
    document.getElementById("champ-header-lineage").textContent = `Champion: ${data.brain.lineage} (Action: ${data.brain.action_name})`;
  }

  // Leaderboard
  const listEl = document.getElementById("leaderboard-list");
  if (listEl && data.agents) {
    const sorted = [...data.agents].sort((a, b) => b.food_eaten - a.food_eaten);
    listEl.innerHTML = sorted.map(a => `
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem;">
        <span style="display: flex; align-items: center; gap: 6px;">
          <span style="display:inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${a.color};"></span>
          <span style="color: ${a.is_champion ? '#facc15' : '#e2e8f0'}; font-weight: ${a.is_champion ? '700' : '400'};">
            ${a.is_champion ? '👑 ' : ''}${a.lineage}
          </span>
        </span>
        <span style="font-weight: 700; color: #facc15;">${a.food_eaten}</span>
      </div>
    `).join("");
  }

  const playBtn = document.getElementById("btn-play-pause");
  playBtn.textContent = data.is_paused ? "▶ Resume" : "⏸ Pause";
}

// URL Routing
function handleRouting() {
  const path = window.location.pathname;
  const hash = window.location.hash.toLowerCase();

  if (path === "/neural" || hash === "#neural") {
    switchView("neural");
  } else if (hash === "#world") {
    switchView("world");
  } else {
    switchView("split");
  }
}

function switchView(view) {
  currentView = view;
  const worldSec = document.getElementById("sec-world");
  const brainSec = document.getElementById("sec-brain");

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });

  if (view === "world") {
    worldSec.classList.remove("hidden");
    brainSec.classList.add("hidden");
    window.location.hash = "world";
  } else if (view === "neural") {
    worldSec.classList.add("hidden");
    brainSec.classList.remove("hidden");
    window.location.hash = "neural";
  } else {
    worldSec.classList.remove("hidden");
    brainSec.classList.remove("hidden");
    window.location.hash = "split";
  }

  resizeCanvases();
  render();
}

function resizeCanvases() {
  const worldContainer = document.getElementById("world-container");
  if (worldContainer) {
    const size = Math.min(worldContainer.clientWidth, worldContainer.clientHeight) - 20;
    if (size > 100) {
      worldCanvas.width = size;
      worldCanvas.height = size;
    }
  }

  const brainContainer = document.getElementById("brain-container");
  if (brainContainer) {
    brainCanvas.width = brainContainer.clientWidth - 20;
    brainCanvas.height = brainContainer.clientHeight - 20;
  }
}

function render() {
  if (currentState) {
    if (!document.getElementById("sec-world").classList.contains("hidden")) {
      renderWorld();
    }
    if (!document.getElementById("sec-brain").classList.contains("hidden")) {
      renderBrain();
    }
  }
}

// Render Population Grid
function renderWorld() {
  const size = currentState.grid_size;
  const cellSize = worldCanvas.width / size;

  worldCtx.clearRect(0, 0, worldCanvas.width, worldCanvas.height);

  // Background Grid
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      worldCtx.fillStyle = (r + c) % 2 === 0 ? "#111827" : "#0f172a";
      worldCtx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);

      worldCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      worldCtx.lineWidth = 0.5;
      worldCtx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }

  // Draw Hazards
  for (const [hr, hc] of currentState.hazards) {
    worldCtx.fillStyle = "rgba(239, 68, 68, 0.25)";
    worldCtx.fillRect(hc * cellSize, hr * cellSize, cellSize, cellSize);

    worldCtx.fillStyle = "#ef4444";
    worldCtx.beginPath();
    worldCtx.arc(hc * cellSize + cellSize / 2, hr * cellSize + cellSize / 2, cellSize / 3, 0, Math.PI * 2);
    worldCtx.fill();

    worldCtx.fillStyle = "#ffffff";
    worldCtx.font = "bold 12px sans-serif";
    worldCtx.textAlign = "center";
    worldCtx.fillText("⚠️", hc * cellSize + cellSize / 2, hr * cellSize + cellSize / 2 + 4);
  }

  // Draw Food Pellets
  for (const [fr, fc] of currentState.foods) {
    worldCtx.fillStyle = "rgba(74, 222, 128, 0.3)";
    worldCtx.beginPath();
    worldCtx.arc(fc * cellSize + cellSize / 2, fr * cellSize + cellSize / 2, cellSize / 2.6, 0, Math.PI * 2);
    worldCtx.fill();

    worldCtx.fillStyle = "#22c55e";
    worldCtx.beginPath();
    worldCtx.arc(fc * cellSize + cellSize / 2, fr * cellSize + cellSize / 2, cellSize / 3.8, 0, Math.PI * 2);
    worldCtx.fill();

    worldCtx.fillStyle = "#ffffff";
    worldCtx.font = "bold 10px sans-serif";
    worldCtx.textAlign = "center";
    worldCtx.fillText("🍏", fc * cellSize + cellSize / 2, fr * cellSize + cellSize / 2 + 3);
  }

  // Draw Agents
  for (const agent of currentState.agents) {
    const cx = agent.c * cellSize + cellSize / 2;
    const cy = agent.r * cellSize + cellSize / 2;
    const radius = cellSize / 3.2;

    // Glowing halo
    worldCtx.beginPath();
    worldCtx.arc(cx, cy, radius * 1.5, 0, Math.PI * 2);
    worldCtx.fillStyle = agent.is_champion ? "rgba(250, 204, 21, 0.4)" : "rgba(255, 255, 255, 0.1)";
    worldCtx.fill();

    // Body
    worldCtx.beginPath();
    worldCtx.arc(cx, cy, radius, 0, Math.PI * 2);
    worldCtx.fillStyle = agent.color;
    worldCtx.fill();
    worldCtx.strokeStyle = agent.is_champion ? "#fde047" : "#ffffff";
    worldCtx.lineWidth = agent.is_champion ? 2.5 : 1.2;
    worldCtx.stroke();

    // Label / Crown
    worldCtx.fillStyle = "#ffffff";
    worldCtx.font = "bold 9px sans-serif";
    worldCtx.textAlign = "center";
    if (agent.is_champion) {
      worldCtx.fillText("👑", cx, cy - radius - 2);
    }
    worldCtx.fillText(`${agent.food_eaten}`, cx, cy + 3);
  }
}

// Render Champion Neural Network
function renderBrain() {
  const w = brainCanvas.width;
  const h = brainCanvas.height;
  brainCtx.clearRect(0, 0, w, h);

  const brain = currentState.brain;
  if (!brain || !brain.layers) return;

  const layers = brain.layers;
  const weights = brain.weights;
  const numLayers = layers.length;

  const colWidth = w / (numLayers + 0.6);
  const layerPositions = [];

  // Compute Node Positions
  for (let l = 0; l < numLayers; l++) {
    const layer = layers[l];
    const numNodes = layer.nodes.length;
    const x = (l + 0.8) * colWidth - (colWidth * 0.3);
    const nodeSpacing = (h - 70) / Math.max(numNodes, 1);
    const startY = 45 + (h - 70 - (numNodes - 1) * nodeSpacing) / 2;

    const nodes = [];
    for (let n = 0; n < numNodes; n++) {
      const y = startY + n * nodeSpacing;
      nodes.push({ x, y, data: layer.nodes[n] });
    }
    layerPositions.push(nodes);
  }

  // Draw Synaptic Weights
  for (let l = 0; l < weights.length; l++) {
    const wMatrix = weights[l];
    const fromLayer = layerPositions[l];
    const toLayer = layerPositions[l + 1];

    for (let t = 0; t < toLayer.length; t++) {
      for (let f = 0; f < fromLayer.length; f++) {
        const weightVal = wMatrix[t] ? wMatrix[t][f] : 0;
        const absW = Math.abs(weightVal);
        if (absW < 0.05) continue;

        brainCtx.beginPath();
        brainCtx.moveTo(fromLayer[f].x, fromLayer[f].y);
        brainCtx.lineTo(toLayer[t].x, toLayer[t].y);

        if (weightVal > 0) {
          brainCtx.strokeStyle = `rgba(56, 189, 248, ${Math.min(0.7, absW * 0.4)})`;
        } else {
          brainCtx.strokeStyle = `rgba(244, 63, 94, ${Math.min(0.7, absW * 0.4)})`;
        }

        brainCtx.lineWidth = Math.min(3.0, 0.5 + absW * 1.0);
        brainCtx.stroke();
      }
    }
  }

  // Draw Neurons
  for (let l = 0; l < numLayers; l++) {
    const layer = layers[l];
    const nodes = layerPositions[l];

    brainCtx.fillStyle = "#9ca3af";
    brainCtx.font = "bold 11px sans-serif";
    brainCtx.textAlign = "center";
    brainCtx.fillText(layer.name, nodes[0].x, 26);

    for (let n = 0; n < nodes.length; n++) {
      const node = nodes[n];
      const val = node.data.val;
      const isChosen = node.data.chosen;
      const radius = isChosen ? 12 : 8;

      // Glow
      const intensity = Math.min(1.0, Math.abs(val));
      const glowGrad = brainCtx.createRadialGradient(node.x, node.y, 2, node.x, node.y, radius * 1.8);
      if (isChosen) {
        glowGrad.addColorStop(0, "rgba(250, 204, 21, 0.9)");
        glowGrad.addColorStop(1, "rgba(250, 204, 21, 0.0)");
      } else if (val >= 0) {
        glowGrad.addColorStop(0, `rgba(74, 222, 128, ${0.3 + intensity * 0.6})`);
        glowGrad.addColorStop(1, "rgba(74, 222, 128, 0.0)");
      } else {
        glowGrad.addColorStop(0, `rgba(244, 63, 94, ${0.3 + intensity * 0.6})`);
        glowGrad.addColorStop(1, "rgba(244, 63, 94, 0.0)");
      }

      brainCtx.fillStyle = glowGrad;
      brainCtx.beginPath();
      brainCtx.arc(node.x, node.y, radius * 1.8, 0, Math.PI * 2);
      brainCtx.fill();

      // Node
      brainCtx.beginPath();
      brainCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      brainCtx.fillStyle = isChosen ? "#fde047" : (val >= 0 ? "#4ade80" : "#f43f5e");
      brainCtx.fill();
      brainCtx.strokeStyle = "#ffffff";
      brainCtx.lineWidth = isChosen ? 2 : 1;
      brainCtx.stroke();

      // Value label
      brainCtx.fillStyle = "#ffffff";
      brainCtx.font = "8px monospace";
      brainCtx.textAlign = "center";
      brainCtx.fillText(val.toFixed(2), node.x, node.y + (radius + 10));

      // Name label
      if (node.data.name) {
        brainCtx.fillStyle = isChosen ? "#facc15" : "#e2e8f0";
        brainCtx.font = isChosen ? "bold 11px sans-serif" : "10px sans-serif";
        if (l === 0) {
          brainCtx.textAlign = "right";
          brainCtx.fillText(node.data.name, node.x - (radius + 6), node.y + 3);
        } else if (l === numLayers - 1) {
          brainCtx.textAlign = "left";
          brainCtx.fillText((isChosen ? "👉 " : "") + node.data.name, node.x + (radius + 6), node.y + 3);
        }
      }
    }
  }
}

// Event Listeners
window.addEventListener("DOMContentLoaded", () => {
  connectWS();
  handleRouting();

  window.addEventListener("resize", () => {
    resizeCanvases();
    render();
  });
  window.addEventListener("hashchange", handleRouting);

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  document.getElementById("btn-play-pause").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ action: "toggle_pause" }));
  });

  document.getElementById("btn-step").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ action: "step_once" }));
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ action: "reset" }));
  });

  document.getElementById("speed-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById("speed-label").textContent = `${val}x`;
    if (ws) ws.send(JSON.stringify({ action: "set_speed", speed: val }));
  });
});
