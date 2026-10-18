// Scarcity & Famine 200-Heroes Frontend Client
let ws = null;
let currentState = null;
let selectedHeroId = 1;
let currentFilter = "all";
let searchQuery = "";

const canvas = document.getElementById("scarcity-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tile-tooltip");

let hoveredTile = null;

// Connect to WebSocket
function connectWS() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/scarcity`);

  ws.onopen = () => {
    const el = document.getElementById("connection-status");
    el.textContent = "Live Connected";
    el.style.color = "var(--accent-emerald)";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "scarcity_update") {
        currentState = data;
        updateUI(data);
      }
    } catch (e) {
      console.error("Failed to parse websocket frame:", e);
    }
  };

  ws.onclose = () => {
    const el = document.getElementById("connection-status");
    el.textContent = "Reconnecting...";
    el.style.color = "var(--accent-rose)";
    setTimeout(connectWS, 1500);
  };
}

// Controls wiring
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
    ws.send(JSON.stringify({ action: "reset", heroes: 200, max_days: 10 }));
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

// Filter buttons
document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderRosterTable();
  });
});

// Search input
document.getElementById("roster-search").addEventListener("input", (e) => {
  searchQuery = e.target.value.trim().toLowerCase();
  renderRosterTable();
});

// Canvas mouse hover & click
canvas.addEventListener("mousemove", (e) => {
  if (!currentState) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mouseX = (e.clientX - rect.left) * scaleX;
  const mouseY = (e.clientY - rect.top) * scaleY;

  const size = currentState.grid_size || 14;
  const cellSize = canvas.width / size;

  const col = Math.floor(mouseX / cellSize);
  const row = Math.floor(mouseY / cellSize);

  if (row >= 0 && row < size && col >= 0 && col < size) {
    hoveredTile = { r: row, c: col };
    showTooltip(row, col, e.clientX, e.clientY);
  } else {
    hoveredTile = null;
    tooltip.style.display = "none";
  }
});

canvas.addEventListener("mouseleave", () => {
  hoveredTile = null;
  tooltip.style.display = "none";
});

canvas.addEventListener("click", () => {
  if (!currentState || !hoveredTile) return;
  const heroesOnTile = currentState.heroes.filter(
    (h) => h.alive && h.r === hoveredTile.r && h.c === hoveredTile.c
  );
  if (heroesOnTile.length > 0) {
    selectedHeroId = heroesOnTile[0].id;
    updateInspector(selectedHeroId);
    renderRosterTable();
  }
});

function showTooltip(row, col, clientX, clientY) {
  if (!currentState) return;
  const isHome = currentState.home_pos[0] === row && currentState.home_pos[1] === col;
  const isWater = currentState.water_tiles.some(([r, c]) => r === row && c === col);
  const isHazard = currentState.hazards.some(([r, c]) => r === row && c === col);
  const bush = currentState.bushes.find((b) => b.pos[0] === row && b.pos[1] === col);
  const tree = currentState.trees.find((t) => t.pos[0] === row && t.pos[1] === col);

  const heroes = currentState.heroes.filter((h) => h.alive && h.r === row && h.c === col);

  let title = `Tile (${row}, ${col})`;
  if (isHome) title += " · ⛺ Home Base & Granary";
  else if (isWater) title += " · 💧 Water Lake";
  else if (isHazard) title += " · ⚠️ Hazard Trap";
  else if (bush) title += bush.is_ripe ? " · 🍏 Ripe Bush" : ` · 🌱 Barren Bush (${bush.timer}t)`;
  else if (tree) title += tree.is_mature ? " · 🌲 Timber Tree" : ` · 🪵 Stump (${tree.timer}t)`;

  let content = `<div style="font-weight:700; border-bottom:1px solid #334466; padding-bottom:4px; margin-bottom:6px;">${title}</div>`;

  if (heroes.length === 0) {
    content += `<div style="color:var(--text-muted);">No living heroes on this tile.</div>`;
  } else {
    content += `<div style="font-weight:600; color:var(--accent-cyan); margin-bottom:4px;">👥 ${heroes.length} Heroes Present:</div>`;
    content += `<div style="max-height:120px; overflow-y:auto; display:flex; flex-direction:column; gap:2px;">`;
    heroes.slice(0, 10).forEach((h) => {
      const hpColor = h.vitals.health > 40 ? "#10b981" : "#ef4444";
      const pack = "🍏".repeat(h.backpack.food) + "🪵".repeat(h.backpack.wood) || "empty";
      content += `<div><b>H${String(h.id).padStart(2, "0")}</b> · <span style="color:${hpColor};">${Math.round(h.vitals.health)}% HP</span> · Pack: ${pack} · <i>${h.goal}</i></div>`;
    });
    if (heroes.length > 10) {
      content += `<div style="color:var(--text-muted); font-style:italic;">+ ${heroes.length - 10} more...</div>`;
    }
    content += `</div>`;
  }

  tooltip.innerHTML = content;
  tooltip.style.display = "block";
  tooltip.style.left = `${clientX + 16}px`;
  tooltip.style.top = `${clientY + 16}px`;
}

// UI Updating
function updateUI(data) {
  // HUD
  document.getElementById("hud-day").textContent = `Day ${data.day_num} / ${data.max_days} · ${data.tod_name}`;
  const survivalPct = data.stats.survival_rate;
  const aliveEl = document.getElementById("hud-alive");
  aliveEl.textContent = `${data.alive_count} / ${data.total_heroes} (${survivalPct.toFixed(1)}%)`;
  aliveEl.style.color = survivalPct > 70 ? "var(--accent-emerald)" : (survivalPct > 40 ? "var(--accent-amber)" : "var(--accent-rose)");

  document.getElementById("hud-granary").textContent = `🍏 ${data.stockpile.food} | 🪵 ${data.stockpile.wood}`;
  document.getElementById("hud-rations").textContent = `${data.emergency_rations} 🍲`;

  const famineBadge = document.getElementById("hud-famine-badge");
  const ripeBushes = data.bushes.filter((b) => b.is_ripe).length;
  if (ripeBushes === 0) {
    famineBadge.className = "badge-famine-critical";
    famineBadge.textContent = "⚠️ CRITICAL FAMINE: 0 APPLES ON MAP";
  } else {
    famineBadge.className = "badge-famine-stable";
    famineBadge.textContent = `🌱 Ecology Stable (${ripeBushes} Ripe Bushes)`;
  }

  // Play/Pause button
  document.getElementById("btn-play-pause").textContent = data.is_paused ? "▶ Play" : "⏸ Pause";

  // Colony Vitals
  const s = data.stats;
  document.getElementById("meter-val-health").textContent = `${Math.round(s.avg_health)}%`;
  document.getElementById("bar-fill-health").style.width = `${s.avg_health}%`;
  document.getElementById("bar-fill-health").style.background = s.avg_health > 40 ? "var(--accent-emerald)" : "var(--accent-rose)";

  document.getElementById("meter-val-energy").textContent = `${Math.round(s.avg_energy)}%`;
  document.getElementById("bar-fill-energy").style.width = `${s.avg_energy}%`;

  document.getElementById("meter-val-hunger").textContent = `${Math.round(s.avg_hunger)}%`;
  document.getElementById("bar-fill-hunger").style.width = `${s.avg_hunger}%`;
  document.getElementById("bar-fill-hunger").style.background = s.avg_hunger >= 70 ? "var(--accent-rose)" : "var(--accent-amber)";

  document.getElementById("meter-val-thirst").textContent = `${Math.round(s.avg_thirst)}%`;
  document.getElementById("bar-fill-thirst").style.width = `${s.avg_thirst}%`;
  document.getElementById("bar-fill-thirst").style.background = s.avg_thirst >= 70 ? "var(--accent-rose)" : "var(--accent-cyan)";

  // Goal Breakdown
  const goals = { FORAGE: 0, HYDRATE: 0, RETURN_HOME: 0, REST: 0, SURVIVE: 0 };
  data.heroes.forEach((h) => {
    if (h.alive && goals[h.goal] !== undefined) {
      goals[h.goal]++;
    }
  });
  document.getElementById("count-forage").textContent = goals.FORAGE;
  document.getElementById("count-hydrate").textContent = goals.HYDRATE;
  document.getElementById("count-return").textContent = goals.RETURN_HOME;
  document.getElementById("count-rest").textContent = goals.REST;
  document.getElementById("count-survive").textContent = goals.SURVIVE;

  // Mortality
  document.getElementById("stat-starve-deaths").textContent = s.starvation_deaths;
  document.getElementById("stat-thirst-deaths").textContent = s.dehydration_deaths;
  document.getElementById("stat-hazard-deaths").textContent = s.hazard_deaths;
  const deadCount = data.total_heroes - data.alive_count;
  const mortPct = (deadCount / Math.max(1, data.total_heroes)) * 100;
  document.getElementById("mortality-rate-tag").textContent = `${mortPct.toFixed(1)}% Mortality (${deadCount} Dead)`;

  // Event feed
  if (data.recent_events && data.recent_events.length > 0) {
    const feedContainer = document.getElementById("event-feed-container");
    feedContainer.innerHTML = "";
    data.recent_events.slice().reverse().forEach((ev) => {
      const line = document.createElement("div");
      line.className = "event-line";
      if (ev.includes("perished")) line.classList.add("death");
      else if (ev.includes("ration")) line.classList.add("ration");
      else if (ev.includes("picked") || ev.includes("chopped")) line.classList.add("harvest");
      line.textContent = `[T${data.tick}] ${ev}`;
      feedContainer.appendChild(line);
    });
  }

  // Roster & Inspector
  updateInspector(selectedHeroId);
  renderRosterTable();
}

function updateInspector(heroId) {
  if (!currentState) return;
  const hero = currentState.heroes.find((h) => h.id === heroId);
  if (!hero) return;

  document.getElementById("insp-name").textContent = `Hero #${String(hero.id).padStart(3, "0")}`;
  const statusEl = document.getElementById("insp-status");
  if (hero.alive) {
    statusEl.textContent = `ALIVE · Survived ${hero.days_survived.toFixed(1)}d`;
    statusEl.style.color = "var(--accent-emerald)";
  } else {
    statusEl.textContent = `FALLEN (${hero.cause_of_death || "Perished"})`;
    statusEl.style.color = "var(--accent-rose)";
  }

  document.getElementById("insp-hp").textContent = `${Math.round(hero.vitals.health)}%`;
  document.getElementById("insp-hp").style.color = hero.vitals.health > 40 ? "var(--accent-emerald)" : "var(--accent-rose)";

  document.getElementById("insp-hunger").textContent = `${Math.round(hero.vitals.hunger)}%`;
  document.getElementById("insp-hunger").style.color = hero.vitals.hunger >= 70 ? "var(--accent-rose)" : "var(--accent-amber)";

  document.getElementById("insp-thirst").textContent = `${Math.round(hero.vitals.thirst)}%`;
  document.getElementById("insp-thirst").style.color = hero.vitals.thirst >= 70 ? "var(--accent-rose)" : "var(--accent-cyan)";

  const packCount = hero.backpack.food + hero.backpack.wood;
  const packStr = "🍏".repeat(hero.backpack.food) + "🪵".repeat(hero.backpack.wood) || "empty";
  document.getElementById("insp-pack").textContent = `${packCount}/5 (${packStr})`;

  document.getElementById("insp-goal").textContent = hero.goal;
  document.getElementById("insp-deposits").textContent = `🍏 ${hero.food_dep || 0} · 🪵 ${hero.wood_dep || 0}`;
  document.getElementById("insp-rations").textContent = `${hero.rations || 0} 🍲`;

  document.getElementById("selected-hero-tag").textContent = `Inspecting H#${String(hero.id).padStart(3, "0")}`;
}

function renderRosterTable() {
  if (!currentState) return;
  const tbody = document.getElementById("roster-tbody");
  tbody.innerHTML = "";

  let list = currentState.heroes.filter((h) => {
    // Filter
    if (currentFilter === "alive" && !h.alive) return false;
    if (currentFilter === "fallen" && h.alive) return false;
    if (currentFilter === "starving" && (!h.alive || h.vitals.hunger < 70)) return false;

    // Search query
    if (searchQuery) {
      const idStr = String(h.id);
      const nameStr = `h${idStr}`;
      if (!idStr.includes(searchQuery) && !nameStr.includes(searchQuery)) return false;
    }
    return true;
  });

  list.forEach((hero) => {
    const tr = document.createElement("tr");
    if (hero.id === selectedHeroId) tr.classList.add("selected");

    const statusBadge = hero.alive
      ? `<span style="color:var(--accent-emerald); font-weight:700;">ALIVE</span>`
      : `<span style="color:var(--accent-rose);">DEAD</span>`;

    const hpColor = hero.vitals.health > 40 ? "#10b981" : "#ef4444";
    const hungerColor = hero.vitals.hunger >= 70 ? "#ef4444" : "#facc15";

    const pack = "🍏".repeat(hero.backpack.food) + "🪵".repeat(hero.backpack.wood) || "░";

    tr.innerHTML = `
      <td><b>#${String(hero.id).padStart(3, "0")}</b></td>
      <td>${statusBadge}</td>
      <td style="color:${hpColor};">${Math.round(hero.vitals.health)}%</td>
      <td style="color:${hungerColor};">${Math.round(hero.vitals.hunger)}%</td>
      <td>${pack}</td>
      <td style="font-size:0.7rem; color:#94a3b8;">${hero.goal}</td>
    `;

    tr.addEventListener("click", () => {
      selectedHeroId = hero.id;
      updateInspector(hero.id);
      renderRosterTable();
    });

    tbody.appendChild(tr);
  });
}

// Canvas Rendering Loop (60 FPS)
let animBeacon = 0;

function renderCanvas() {
  requestAnimationFrame(renderCanvas);
  animBeacon = (animBeacon + 0.05) % (Math.PI * 2);

  if (!currentState) return;

  const w = canvas.width;
  const h = canvas.height;
  const size = currentState.grid_size || 14;
  const cellSize = w / size;

  ctx.clearRect(0, 0, w, h);

  // 1. Draw Grid Tiles
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const x = c * cellSize;
      const y = r * cellSize;

      const isWater = currentState.water_tiles.some(([tr, tc]) => tr === r && tc === c);
      const isHome = currentState.home_pos[0] === r && currentState.home_pos[1] === c;
      const isHazard = currentState.hazards.some(([hr, hc]) => hr === r && hc === c);

      // Base background
      if (isWater) {
        ctx.fillStyle = "#0369a1";
        ctx.fillRect(x, y, cellSize, cellSize);
        // Ripple accent
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
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
      } else if (isHazard) {
        ctx.fillStyle = "#450a0a";
        ctx.fillRect(x, y, cellSize, cellSize);
      } else {
        ctx.fillStyle = (r + c) % 2 === 0 ? "#0f172a" : "#111b33";
        ctx.fillRect(x, y, cellSize, cellSize);
      }

      // Tile borders
      ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellSize, cellSize);

      // Landmarks & Ecological Features
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
      } else if (isHazard) {
        ctx.font = `${cellSize * 0.35}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("⚠️", x + cellSize * 0.5, y + cellSize * 0.5);
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
          // Barren bush with cooldown progress ring
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
          // Stump with cooldown progress ring
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

  // 2. Aggregate and Render Living Heroes by Cell
  const cellMap = new Map();
  currentState.heroes.forEach((h) => {
    if (!h.alive) return;
    const key = `${h.r},${h.c}`;
    if (!cellMap.has(key)) cellMap.set(key, []);
    cellMap.get(key).push(h);
  });

  cellMap.forEach((heroes, key) => {
    const [r, c] = key.split(",").map(Number);
    const x = c * cellSize;
    const y = r * cellSize;
    const count = heroes.length;

    if (count === 1) {
      const hero = heroes[0];
      const isSelected = hero.id === selectedHeroId;

      // Single Hero Marker
      const cx = x + cellSize * 0.5;
      const cy = y + cellSize * 0.5;
      const radius = cellSize * 0.32;

      // Health ring
      ctx.strokeStyle = hero.vitals.health > 40 ? "#10b981" : "#ef4444";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Body fill
      ctx.fillStyle = hero.goal === "REST" ? "#38bdf8" : (hero.goal === "HYDRATE" ? "#06b6d4" : "#facc15");
      ctx.beginPath();
      ctx.arc(cx, cy, radius - 2, 0, Math.PI * 2);
      ctx.fill();

      // Icon
      ctx.font = `${cellSize * 0.32}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(hero.goal === "REST" ? "💤" : "🧙", cx, cy);

    } else {
      // Multi-Hero Cluster Badge
      const cx = x + cellSize * 0.5;
      const cy = y + cellSize * 0.5;
      const bw = cellSize * 0.72;
      const bh = cellSize * 0.44;

      // Majority state color
      const avgHp = heroes.reduce((acc, h) => acc + h.vitals.health, 0) / count;
      const borderColor = avgHp > 40 ? "#38bdf8" : "#f43f5e";

      ctx.fillStyle = "rgba(18, 24, 38, 0.92)";
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(cx - bw / 2, cy - bh / 2, bw, bh, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${cellSize * 0.28}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`👥${count}`, cx, cy);
    }
  });

  // 3. Highlight Selected Hero Beacon
  const selectedHero = currentState.heroes.find((h) => h.id === selectedHeroId);
  if (selectedHero && selectedHero.alive) {
    const sx = selectedHero.c * cellSize + cellSize * 0.5;
    const sy = selectedHero.r * cellSize + cellSize * 0.5;
    const beaconRadius = cellSize * 0.45 + Math.sin(animBeacon) * (cellSize * 0.12);

    ctx.strokeStyle = "rgba(6, 182, 212, 0.85)";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(4, beaconRadius), 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Beacon label
    ctx.fillStyle = "#06b6d4";
    ctx.font = "bold 10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`H#${String(selectedHero.id).padStart(3, "0")}`, sx, sy - cellSize * 0.55);
  }

  // 4. Highlight Hovered Tile
  if (hoveredTile) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
    ctx.lineWidth = 2;
    ctx.strokeRect(hoveredTile.c * cellSize, hoveredTile.r * cellSize, cellSize, cellSize);
  }
}

// Start WebSocket and Canvas Loop
connectWS();
requestAnimationFrame(renderCanvas);
