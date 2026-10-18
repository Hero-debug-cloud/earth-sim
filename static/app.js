// Unified Civilization Platform Client
let ws = null;
let currentState = null;
let selectedHeroId = 1;
let animBeacon = 0;

const canvas = document.getElementById("world-canvas");
const ctx = canvas.getContext("2d");

// Connect WebSocket
function connectWS() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    document.getElementById("connection-status").textContent = "Live Connected";
    document.getElementById("connection-status").style.color = "var(--accent-emerald)";
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
    document.getElementById("connection-status").textContent = "Reconnecting...";
    document.getElementById("connection-status").style.color = "var(--accent-rose)";
    setTimeout(connectWS, 1500);
  };
}

// Evolution & Controls Wiring
document.getElementById("btn-evolve").addEventListener("click", () => {
  const inputEl = document.getElementById("evolve-gens-input");
  const gens = parseInt(inputEl.value) || 10;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "evolve_generations", generations: gens }));
  }
});

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
speedSlider.addEventListener("input", (e) => {
  const spd = parseFloat(e.target.value);
  document.getElementById("speed-label").textContent = `${spd}x`;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "set_speed", speed: spd }));
  }
});

// Sidebar Tab Switching
document.querySelectorAll(".sidebar-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".sidebar-tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}-content`).classList.add("active");
  });
});

// Canvas Click to Select Hero
canvas.addEventListener("click", (e) => {
  if (!currentState) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  const size = currentState.grid_size || 14;
  const cellSize = canvas.width / size;
  const col = Math.floor(clickX / cellSize);
  const row = Math.floor(clickY / cellSize);

  const hero = currentState.heroes.find((h) => h.r === row && h.c === col);
  if (hero) {
    selectedHeroId = hero.id;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "select_hero", id: hero.id }));
    }
  }
});

function updateUI(data) {
  // Status Bar
  document.getElementById("stat-gen").textContent = data.generation;
  document.getElementById("stat-time").textContent = `Day ${data.day_num} · ${data.tod_name}`;
  document.getElementById("stat-granary").textContent = `🍏 ${data.stockpile.food} | 🪵 ${data.stockpile.wood}`;
  document.getElementById("stat-inspecting").textContent = `Hero #${selectedHeroId}`;

  // Evolution Indicator
  const evoBadge = document.getElementById("evolution-indicator");
  const btnEvolve = document.getElementById("btn-evolve");
  if (data.is_evolving) {
    evoBadge.style.display = "block";
    btnEvolve.disabled = true;
    btnEvolve.textContent = "Adapting...";
  } else {
    evoBadge.style.display = "none";
    btnEvolve.disabled = false;
    btnEvolve.textContent = "🧬 Evolve";
  }

  // Play/Pause button
  document.getElementById("btn-play-pause").textContent = data.is_paused ? "▶ Play" : "⏸ Pause";

  // Selected Hero Vitals
  const inspectedHero = data.heroes.find((h) => h.id === selectedHeroId) || data.heroes[0];
  if (inspectedHero) {
    document.getElementById("hero-name-label").textContent = `Hero #${inspectedHero.id} Vitals`;
    const goalBadge = document.getElementById("hero-goal-badge");
    goalBadge.textContent = inspectedHero.goal;

    const v = inspectedHero.vitals;
    document.getElementById("val-health").textContent = `${Math.round(v.health)}%`;
    document.getElementById("bar-health").style.width = `${v.health}%`;
    document.getElementById("bar-health").style.background = v.health > 40 ? "var(--accent-emerald)" : "var(--accent-rose)";

    document.getElementById("val-energy").textContent = `${Math.round(v.energy)}%`;
    document.getElementById("bar-energy").style.width = `${v.energy}%`;

    document.getElementById("val-hunger").textContent = `${Math.round(v.hunger)}%`;
    document.getElementById("bar-hunger").style.width = `${v.hunger}%`;
    document.getElementById("bar-hunger").style.background = v.hunger >= 70 ? "var(--accent-rose)" : "var(--accent-amber)";

    document.getElementById("val-thirst").textContent = `${Math.round(v.thirst)}%`;
    document.getElementById("bar-thirst").style.width = `${v.thirst}%`;
    document.getElementById("bar-thirst").style.background = v.thirst >= 70 ? "var(--accent-rose)" : "var(--accent-cyan)";

    // Backpack
    const bp = inspectedHero.backpack;
    const bpEl = document.getElementById("backpack-slots-container");
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
        slot.textContent = "░";
      }
      bpEl.appendChild(slot);
    }
  }

  // Real-Time Neural Brain Viewer
  if (data.brain) {
    const b = data.brain;
    document.getElementById("lineage-tag").textContent = b.lineage;

    // Sensors
    const sensorContainer = document.getElementById("nodes-sensor");
    sensorContainer.innerHTML = "";
    b.sensors.forEach((s) => {
      const node = document.createElement("div");
      node.className = "neural-node";
      node.innerHTML = `
        <div class="neural-node-row"><span>${s.name}</span><span class="neural-node-val">${s.val}</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.abs(s.val) * 100}%; background:var(--accent-cyan);"></div></div>
      `;
      sensorContainer.appendChild(node);
    });

    // Hidden GRU State
    const hiddenContainer = document.getElementById("nodes-hidden");
    hiddenContainer.innerHTML = "";
    b.hidden.forEach((h) => {
      const norm = (h.val + 1.0) / 2.0; // [-1, 1] to [0, 1]
      const node = document.createElement("div");
      node.className = "neural-node";
      node.innerHTML = `
        <div class="neural-node-row"><span>${h.name}</span><span class="neural-node-val">${h.val}</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width:${norm * 100}%; background:var(--accent-purple);"></div></div>
      `;
      hiddenContainer.appendChild(node);
    });

    // Motor Actions
    const actionContainer = document.getElementById("nodes-action");
    actionContainer.innerHTML = "";
    b.actions.forEach((a) => {
      const node = document.createElement("div");
      node.className = `neural-node ${a.chosen ? "chosen" : ""}`;
      node.innerHTML = `
        <div class="neural-node-row"><span>${a.name}</span><span class="neural-node-val">${Math.round(a.val * 100)}%</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width:${a.val * 100}%; background:${a.chosen ? "var(--accent-emerald)" : "var(--accent-cyan)"};"></div></div>
      `;
      actionContainer.appendChild(node);
    });
  }

  // Colony Stats
  document.getElementById("colony-food").textContent = data.stockpile.food;
  document.getElementById("colony-wood").textContent = data.stockpile.wood;
  document.getElementById("colony-rations").textContent = data.emergency_rations;

  // Roster
  const rosterContainer = document.getElementById("roster-container");
  rosterContainer.innerHTML = "";
  data.heroes.forEach((h) => {
    const item = document.createElement("div");
    item.className = `roster-item ${h.id === selectedHeroId ? "active" : ""}`;
    const pack = "🍏".repeat(h.backpack.food) + "🪵".repeat(h.backpack.wood) || "empty";
    item.innerHTML = `
      <span><b>Hero #${h.id}</b> · ${Math.round(h.vitals.health)}% HP</span>
      <span style="color:var(--text-secondary); font-size:0.7rem;">Pack: ${pack} · <i>${h.goal}</i></span>
    `;
    item.addEventListener("click", () => {
      selectedHeroId = h.id;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "select_hero", id: h.id }));
      }
    });
    rosterContainer.appendChild(item);
  });

  // Event Feed
  if (data.events && data.events.length > 0) {
    const feed = document.getElementById("event-feed");
    feed.innerHTML = "";
    data.events.slice().reverse().forEach((ev) => {
      const line = document.createElement("div");
      line.textContent = `[T${data.tick}] ${ev}`;
      feed.appendChild(line);
    });
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

      // Bushes
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

      // Trees
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

  // 3. Render Heroes
  currentState.heroes.forEach((h) => {
    const x = h.c * cellSize + cellSize * 0.5;
    const y = h.r * cellSize + cellSize * 0.5;
    const radius = cellSize * 0.32;

    // Selected Beacon
    if (h.id === selectedHeroId) {
      const beaconR = cellSize * 0.45 + Math.sin(animBeacon) * (cellSize * 0.1);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.9)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(4, beaconR), 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

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
  });
}

connectWS();
requestAnimationFrame(render);
