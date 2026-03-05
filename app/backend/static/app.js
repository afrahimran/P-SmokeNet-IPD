async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

let chart;
let seriesGlobal = [];
let activeIndex = -1;

function nearestIndexByTime(series, t) {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < series.length; i++) {
    const d = Math.abs(series[i].t - t);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function setActive(i) {
  activeIndex = i;
  const p = seriesGlobal[i];
  if (!p) return;

  document.getElementById("out").textContent = JSON.stringify(p, null, 2);
  document.getElementById("curP").textContent = p.prob.toFixed(4);

  // highlight point via point radius trick
  if (chart) {
    const base = 2;
    const radii = seriesGlobal.map((_, idx) => idx === i ? 6 : base);
    chart.data.datasets[0].pointRadius = radii;
    chart.update("none");
  }
}

function renderChart(series) {
  const labels = series.map(p => p.t);
  const probs = series.map(p => p.prob);

  const gtPoints = series.map((p, i) => (p.ground_truth === 1 ? probs[i] : null));
  const flagged = series.map((p, i) => (p.decision === 1 ? probs[i] : null));

  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Predicted P(smoke onset)", data: probs, tension: 0.25, pointRadius: 2 },
        { label: "Ground truth positive clips", data: gtPoints, showLine: false, pointRadius: 4 },
        { label: "Flagged (>= threshold)", data: flagged, showLine: false, pointRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } },
      scales: {
        y: { min: 0, max: 1, title: { display: true, text: "Probability" } },
        x: { title: { display: true, text: "Time (sec)" } }
      },
      onClick: (evt, elements) => {
        if (!elements.length) return;
        const idx = elements[0].index;
        setActive(idx);
      }
    }
  });
}

async function init() {
  const videoSelect = document.getElementById("videoSelect");
  const status = document.getElementById("status");
  const videoStatus = document.getElementById("videoStatus");
  const vid = document.getElementById("vid");
  const vidSrc = document.getElementById("vidSrc");

  const vids = await getJSON("/api/videos?split=test");
  vids.videos.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    videoSelect.appendChild(opt);
  });

  if (vids.videos.includes("video04")) videoSelect.value = "video04";

  function loadVideoForSelected() {
    const v = videoSelect.value;
    vidSrc.src = `/videos/${encodeURIComponent(v)}.mp4`;
    vid.load();
    videoStatus.textContent = `Loaded: ${v}.mp4`;
  }

  loadVideoForSelected();

  videoSelect.addEventListener("change", () => {
    seriesGlobal = [];
    activeIndex = -1;
    status.textContent = "";
    document.getElementById("out").textContent = "";
    document.getElementById("curP").textContent = "-";
    loadVideoForSelected();
  });

  vid.addEventListener("timeupdate", () => {
    document.getElementById("curT").textContent = vid.currentTime.toFixed(1);
    if (!seriesGlobal.length) return;
    const idx = nearestIndexByTime(seriesGlobal, vid.currentTime);
    if (idx !== activeIndex) setActive(idx);
  });

  document.getElementById("runBtn").addEventListener("click", async () => {
    status.textContent = "Running inference...";
    const v = videoSelect.value;
    const tStart = document.getElementById("tStart").value;
    const tEnd = document.getElementById("tEnd").value;
    const stepS = document.getElementById("stepS").value;
    const thr = document.getElementById("thr").value;

    const data = await getJSON(`/api/predict_series?video_id=${encodeURIComponent(v)}&t_start=${tStart}&t_end=${tEnd}&step_s=${stepS}&threshold=${thr}`);
    seriesGlobal = data.series;
    status.textContent = `Points: ${seriesGlobal.length} | threshold=${data.threshold}`;

    renderChart(seriesGlobal);

    vid.currentTime = Number(tStart);
    setActive(nearestIndexByTime(seriesGlobal, Number(tStart)));
  });
}

init().catch(err => {
  document.getElementById("status").textContent = "Error";
  document.getElementById("out").textContent = String(err);
});