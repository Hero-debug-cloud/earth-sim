// Unified Living World Client · Single Champion Hero
let ws = null;
let currentState = null;
let animBeacon = 0;

const canvas = document.getElementById("world-canvas");
const ctx = canvas.getContext("2d");

// Connect WebSocket
function connectWS() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    const statusEl = document.getElementById("connection-status");
    if (statusEl) {
      statusEl.textContent = "Live Connected";
      statusEl.style.color = "var(--accent-emerald)";
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "world_update") {
        currentState = data;
        updateUI(data);
      }
    } catch (e) {
      console.error("Frame parse error:", e);
    }
  };

  ws.onclose = () => {
    const statusEl = document.getElementById("connection-status");
    if (statusEl) {
      statusEl.textContent = "Reconnecting...";
      statusEl.style.color = "var(--accent-rose)";
    }
    setTimeout(connectWS, 1500);
  };
}

// Evolution & Controls Wiring
const btnEvolve = document.getElementById("btn-evolve");
if (btnEvolve) {
  btnEvolve.addEventListener("click", () => {
    const inputEl = document.getElementById("evolve-gens-input");
    const gens = parseInt(inputEl.value) || 10;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "evolve_generations", generations: gens }));
    }
  });
}

const btnPlayPause = document.getElementById("btn-play-pause");
if (btnPlayPause) {
  btnPlayPause.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "toggle_pause" }));
    }
  });
}

const btnStep = document.getElementById("btn-step");
if (btnStep) {
  btnStep.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "step_once" }));
    }
  });
}

const btnReset = document.getElementById("btn-reset");
if (btnReset) {
  btnReset.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "reset" }));
    }
  });
}

const speedSlider = document.getElementById("speed-slider");
if (speedSlider) {
  speedSlider.addEventListener("input", (e) => {
    const spd = parseFloat(e.target.value);
    const lbl = document.getElementById("speed-label");
    if (lbl) lbl.textContent = `${spd}x`;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "set_speed", speed: spd }));
    }
  });
}

function updateUI(data) {
  if (!data) return;

  // Status Bar
  const statGen = document.getElementById("stat-gen");
  if (statGen) statGen.textContent = data.generation;

  const statTime = document.getElementById("stat-time");
  if (statTime) statTime.textContent = `Day ${data.day_num} · ${data.tod_name}`;

  const statGranary = document.getElementById("stat-granary");
  if (statGranary) statGranary.textContent = `🍏 ${data.stockpile.food} | 🪵 ${data.stockpile.wood}`;

  // Evolution Indicator
  const evoBadge = document.getElementById("evolution-indicator");
  const evolveBtn = document.getElementById("btn-evolve");
  if (evoBadge && evolveBtn) {
    if (data.is_evolving) {
      evoBadge.style.display = "block";
      evolveBtn.disabled = true;
      evolveBtn.textContent = "Adapting...";
    } else {
      evoBadge.style.display = "none";
      evolveBtn.disabled = false;
      evolveBtn.textContent = "🧬 Evolve";
    }
  }

  // Play/Pause button
  const playPauseBtn = document.getElementById("btn-play-pause");
  if (playPauseBtn) {
    playPauseBtn.textContent = data.is_paused ? "▶ Play" : "⏸ Pause";
  }

  // Day Cycle Meter in Bottom Ribbon
  const dayCycleFill = document.getElementById("day-cycle-fill");
  if (dayCycleFill) {
    const cycleProgress = data.tick % 100;
    dayCycleFill.style.width = `${cycleProgress}%`;
  }

  // Single Champion Hero Vitals & Homeostasis
  const championHero = (data.heroes && data.heroes.length > 0) ? data.heroes[0] : null;
  if (championHero) {
    const nameLabel = document.getElementById("hero-name-label");
    if (nameLabel) nameLabel.textContent = `Champion Homeostasis`;

    const lineageEl = document.getElementById("lineage-tag");
    if (lineageEl) lineageEl.textContent = championHero.lineage || `Gen ${data.generation} Champion`;

    const goalBadge = document.getElementById("hero-goal-badge");
    if (goalBadge) {
      goalBadge.textContent = championHero.goal;
      if (championHero.goal === "REST") {
        goalBadge.style.borderColor = "var(--accent-cyan)";
        goalBadge.style.color = "var(--accent-cyan)";
        goalBadge.style.background = "rgba(6, 182, 212, 0.15)";
      } else if (championHero.goal === "HYDRATE") {
        goalBadge.style.borderColor = "#38bdf8";
        goalBadge.style.color = "#38bdf8";
        goalBadge.style.background = "rgba(56, 189, 248, 0.15)";
      } else {
        goalBadge.style.borderColor = "var(--accent-emerald)";
        goalBadge.style.color = "var(--accent-emerald)";
        goalBadge.style.background = "rgba(16, 185, 129, 0.15)";
      }
    }

    const v = championHero.vitals;
    const valHealth = document.getElementById("val-health");
    const barHealth = document.getElementById("bar-health");
    if (valHealth && barHealth) {
      valHealth.textContent = `${Math.round(v.health)}%`;
      barHealth.style.width = `${v.health}%`;
      barHealth.style.background = v.health > 40 ? "var(--accent-emerald)" : "var(--accent-rose)";
    }

    const valEnergy = document.getElementById("val-energy");
    const barEnergy = document.getElementById("bar-energy");
    if (valEnergy && barEnergy) {
      valEnergy.textContent = `${Math.round(v.energy)}%`;
      barEnergy.style.width = `${v.energy}%`;
    }

    const valHunger = document.getElementById("val-hunger");
    const barHunger = document.getElementById("bar-hunger");
    if (valHunger && barHunger) {
      valHunger.textContent = `${Math.round(v.hunger)}%`;
      barHunger.style.width = `${v.hunger}%`;
      barHunger.style.background = v.hunger >= 70 ? "var(--accent-rose)" : "var(--accent-amber)";
    }

    const valThirst = document.getElementById("val-thirst");
    const barThirst = document.getElementById("bar-thirst");
    if (valThirst && barThirst) {
      valThirst.textContent = `${Math.round(v.thirst)}%`;
      barThirst.style.width = `${v.thirst}%`;
      barThirst.style.background = v.thirst >= 70 ? "var(--accent-rose)" : "var(--accent-cyan)";
    }

    // 5-Slot Backpack
    const bp = championHero.backpack;
    const bpEl = document.getElementById("backpack-slots-container");
    if (bpEl) {
      bpEl.innerHTML = "";
      for (let i = 0; i < 5; i++) {
        const slot = document.createElement("div");
        slot.className = "backpack-slot";
        if (i < bp.food) {
          slot.classList.add("filled");
          slot.textContent = "🍏";
        } else if (i < bp.food + bp.wood) {
          slot.classList.add("filled");
          slot.textContent = "🪵";
        } else {
          slot.textContent = "";
        }
        bpEl.appendChild(slot);
      }
    }
  }

  // Real-Time Neural Core Viewer
  if (data.brain) {
    const b = data.brain;

    // 1. Sensory Inputs (6D)
    const sensorContainer = document.getElementById("nodes-sensor");
    if (sensorContainer) {
      sensorContainer.innerHTML = "";
      b.sensors.forEach((s) => {
        const row = document.createElement("div");
        row.className = "sensor-row";
        const val = s.val;
        let fillLeft = 50;
        let fillWidth = 0;
        let fillColor = "var(--accent-cyan)";
        if (val < 0) {
          fillWidth = Math.min(50, Math.abs(val) * 50);
          fillLeft = 50 - fillWidth;
          fillColor = "var(--accent-rose)";
        } else {
          fillWidth = Math.min(50, val * 50);
          fillLeft = 50;
          fillColor = "var(--accent-cyan)";
        }

        row.innerHTML = `
          <div class="sensor-header">
            <span>${s.name}</span>
            <span class="sensor-val" style="color: ${fillColor}">${val >= 0 ? "+" : ""}${val.toFixed(2)}</span>
          </div>
          <div class="bipolar-bar">
            <div class="bipolar-fill" style="left: ${fillLeft}%; width: ${fillWidth}%; background: ${fillColor};"></div>
          </div>
        `;
        sensorContainer.appendChild(row);
      });
    }

    // 2. 16D GRU Hidden State Strictly 4x4 Heatmap Matrix (Zero Cutoff)
    const matrixContainer = document.getElementById("matrix-hidden");
    if (matrixContainer) {
      matrixContainer.innerHTML = "";
      b.hidden.forEach((h, idx) => {
        const val = h.val;
        const absVal = Math.min(1.0, Math.abs(val));
        const alpha = Math.max(0.12, absVal * 0.85);
        const bg = val >= 0
          ? `rgba(168, 85, 247, ${alpha.toFixed(2)})`
          : `rgba(239, 68, 68, ${alpha.toFixed(2)})`;
        const border = val >= 0
          ? `rgba(168, 85, 247, ${Math.min(1.0, alpha + 0.3)})`
          : `rgba(239, 68, 68, ${Math.min(1.0, alpha + 0.3)})`;

        const cell = document.createElement("div");
        cell.className = "matrix-cell";
        cell.style.background = bg;
        cell.style.borderColor = border;
        cell.title = `GRU Dimension #${idx}: ${val.toFixed(3)}`;
        cell.innerHTML = `
          <span style="font-size:0.52rem; opacity:0.65;">h${idx}</span>
          <span style="font-weight:700; font-size:0.62rem;">${val >= 0 ? "+" : ""}${val.toFixed(1)}</span>
        `;
        matrixContainer.appendChild(cell);
      });
    }

    // 3. Motor Actions (4D) with bright emerald highlight on chosen action
    const actionContainer = document.getElementById("nodes-action");
    if (actionContainer) {
      actionContainer.innerHTML = "";
      b.actions.forEach((a) => {
        const row = document.createElement("div");
        row.className = `action-row ${a.chosen ? "chosen" : ""}`;
        row.innerHTML = `
          <div class="action-header">
            <span>${a.name}</span>
            <span>${Math.round(a.val * 100)}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:${a.val * 100}%; background:${a.chosen ? "var(--accent-emerald)" : "var(--accent-cyan)"};"></div>
          </div>
        `;
        actionContainer.appendChild(row);
      });
    }
  }

  // Colony Stockpile
  const cf = document.getElementById("colony-food");
  if (cf) cf.textContent = data.stockpile.food;
  const cw = document.getElementById("colony-wood");
  if (cw) cw.textContent = data.stockpile.wood;
  const cr = document.getElementById("colony-rations");
  if (cr) cr.textContent = data.emergency_rations;

  // Live Adventure Chronology (Last 4 events)
  if (data.events && data.events.length > 0) {
    const feed = document.getElementById("event-feed");
    if (feed) {
      feed.innerHTML = "";
      data.events.slice(-4).reverse().forEach((ev) => {
        const line = document.createElement("div");
        line.style.whiteSpace = "nowrap";
        line.style.overflow = "hidden";
        line.style.textOverflow = "ellipsis";
        line.textContent = `[T${data.tick}] ${ev}`;
        feed.appendChild(line);
      });
    }
  }
}

// Canvas Rendering Loop (60 FPS)
function render() {
  requestAnimationFrame(render);
  animBeacon = (animBeacon + 0.05) % (Math.PI * 2);

  if (!currentState) return;

  const w = canvas.width;
  const h = canvas.height;
  const size = currentState.grid_size || 14;
  const cellSize = w / size;

  ctx.clearRect(0, 0, w, h);

  // 1. Grid Tiles & Landmarks
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const x = c * cellSize;
      const y = r * cellSize;

      const isWater = currentState.water_tiles.some(([tr, tc]) => tr === r && tc === c);
      const isHome = currentState.home_pos[0] === r && currentState.home_pos[1] === c;

      if (isWater) {
        ctx.fillStyle = "#0369a1";
        ctx.fillRect(x, y, cellSize, cellSize);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x + cellSize * 0.5, y + cellSize * 0.5, cellSize * 0.3, 0, Math.PI * 2);
        ctx.stroke();
      } else if (isHome) {
        ctx.fillStyle = "#78350f";
        ctx.fillRect(x, y, cellSize, cellSize);
        ctx.strokeStyle = "#b45309";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
      } else {
        ctx.fillStyle = (r + c) % 2 === 0 ? "#0f172a" : "#111b33";
        ctx.fillRect(x, y, cellSize, cellSize);
      }

      ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);

      if (isHome) {
        ctx.font = `${cellSize * 0.45}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("⛺", x + cellSize * 0.5, y + cellSize * 0.45);
      } else if (isWater) {
        ctx.font = `${cellSize * 0.35}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("💧", x + cellSize * 0.5, y + cellSize * 0.5);
      }

      // Berry Bushes
      const bush = currentState.bushes.find((b) => b.pos[0] === r && b.pos[1] === c);
      if (bush) {
        if (bush.is_ripe) {
          ctx.font = `${cellSize * 0.45}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("🍏", x + cellSize * 0.5, y + cellSize * 0.5);
        } else {
          ctx.font = `${cellSize * 0.35}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("🌱", x + cellSize * 0.5, y + cellSize * 0.5);

          const progress = 1.0 - (bush.timer / Math.max(1, bush.max_timer));
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x + cellSize * 0.5, y + cellSize * 0.5, cellSize * 0.38, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
          ctx.stroke();
        }
      }

      // Timber Trees
      const tree = currentState.trees.find((t) => t.pos[0] === r && t.pos[1] === c);
      if (tree) {
        if (tree.is_mature) {
          ctx.font = `${cellSize * 0.45}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("🌲", x + cellSize * 0.5, y + cellSize * 0.5);
        } else {
          ctx.font = `${cellSize * 0.35}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("🪵", x + cellSize * 0.5, y + cellSize * 0.5);

          const progress = 1.0 - (tree.timer / Math.max(1, tree.max_timer));
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x + cellSize * 0.5, y + cellSize * 0.5, cellSize * 0.38, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  }

  // 2. Render Moving Hazards (Predators/Beasts)
  currentState.hazards.forEach(([hr, hc]) => {
    const x = hc * cellSize + cellSize * 0.5;
    const y = hr * cellSize + cellSize * 0.5;

    // Red aura
    ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
    ctx.beginPath();
    ctx.arc(x, y, cellSize * 0.42, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, cellSize * 0.36, 0, Math.PI * 2);
    ctx.stroke();

    ctx.font = `${cellSize * 0.4}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("⚠️", x, y);
  });

  // 3. Render the Champion Hero
  if (currentState.heroes && currentState.heroes.length > 0) {
    const h = currentState.heroes[0];
    const x = h.c * cellSize + cellSize * 0.5;
    const y = h.r * cellSize + cellSize * 0.5;
    const radius = cellSize * 0.32;

    // Active Beacon Ring
    const beaconR = cellSize * 0.45 + Math.sin(animBeacon) * (cellSize * 0.1);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.9)";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(4, beaconR), 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Health ring
    ctx.strokeStyle = h.vitals.health > 40 ? "#10b981" : "#ef4444";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Body
    ctx.fillStyle = h.goal === "REST" ? "#38bdf8" : (h.goal === "HYDRATE" ? "#06b6d4" : "#facc15");
    ctx.beginPath();
    ctx.arc(x, y, radius - 2, 0, Math.PI * 2);
    ctx.fill();

    // Icon
    ctx.font = `${cellSize * 0.32}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(h.goal === "REST" ? "💤" : "🧙", x, y);
  }
}

connectWS();
requestAnimationFrame(render);
