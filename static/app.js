const video = document.querySelector("#video");
const canvas = document.querySelector("#captureCanvas");
const capturePanel = document.querySelector(".capture-panel");
const statusPill = document.querySelector("#statusPill");
const startCameraButton = document.querySelector("#startCamera");
const captureButton = document.querySelector("#captureButton");
const uploadButton = document.querySelector("#uploadButton");
const imageUpload = document.querySelector("#imageUpload");
const analyzeButton = document.querySelector("#analyzeButton");
const solveButton = document.querySelector("#solveButton");
const algorithmSelect = document.querySelector("#algorithmSelect");
const detectedGrid = document.querySelector("#detectedGrid");
const validationBox = document.querySelector("#validationBox");
const outsideValue = document.querySelector("#outsideValue");
const movesBox = document.querySelector("#moves");
const stepsBox = document.querySelector("#steps");
const solveStats = document.querySelector("#solveStats");
const solveProgress = document.querySelector("#solveProgress");
const progressFill = document.querySelector("#progressFill");
const progressText = document.querySelector("#progressText");
const installButton = document.querySelector("#installButton");
const cellTemplate = document.querySelector("#cellTemplate");

window.PUZZLE_LABELS = window.PUZZLE_LABELS || window.PuzzleCore?.allLabels || [
  "00",
  "11", "12", "13", "14",
  "21", "22", "23", "24",
  "31", "32", "33", "34",
  "41", "42", "43", "44",
];
window.PUZZLE_ALGORITHMS = window.PUZZLE_ALGORITHMS || [
  { value: "fast", label: "Fast solver (default)" },
  { value: "auto", label: "Auto + exact fallback" },
  { value: "ida-star", label: "IDA* (exact, slower)" },
  { value: "a-star-closed", label: "A* (exact, memory heavy)" },
  { value: "bfs", label: "BFS (tiny only)" },
];

let stream = null;
let capturedDataUrl = null;
let detectedLabels = Array(16).fill("00");
let activeSolveJobId = null;
let deferredInstallPrompt = null;

function setStatus(text) {
  statusPill.textContent = text;
}

function setValidation(text, tone = "") {
  validationBox.textContent = text;
  validationBox.className = `validation ${tone}`.trim();
}

function setSolveProgress({ visible = true, running = false, progress = 0, text = "waiting" } = {}) {
  solveProgress.hidden = !visible;
  solveProgress.classList.toggle("is-running", running);
  progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  progressText.textContent = text;
}

function resetSolution() {
  activeSolveJobId = null;
  solveButton.disabled = true;
  movesBox.innerHTML = "";
  stepsBox.innerHTML = "";
  solveStats.textContent = "waiting";
  setSolveProgress({ visible: false });
}

function drawDataUrlToCanvas(dataUrl, mode = "cover") {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const side = Math.max(480, Math.min(1200, Math.max(img.naturalWidth, img.naturalHeight)));
      canvas.width = side;
      canvas.height = side;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#0f0e0a";
      ctx.fillRect(0, 0, side, side);

      if (mode === "contain") {
        const scale = Math.min(side / img.naturalWidth, side / img.naturalHeight);
        const dw = img.naturalWidth * scale;
        const dh = img.naturalHeight * scale;
        ctx.drawImage(img, (side - dw) / 2, (side - dh) / 2, dw, dh);
      } else {
        const sourceSide = Math.min(img.naturalWidth, img.naturalHeight);
        const sx = (img.naturalWidth - sourceSide) / 2;
        const sy = (img.naturalHeight - sourceSide) / 2;
        ctx.drawImage(img, sx, sy, sourceSide, sourceSide, 0, 0, side, side);
      }

      capturePanel.classList.add("has-capture");
      resolve();
    };
    img.onerror = () => reject(new Error("image preview could not load"));
    img.src = dataUrl;
  });
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 1280 },
      },
      audio: false,
    });
    video.srcObject = stream;
    capturePanel.classList.remove("has-capture");
    setStatus("camera live");
  } catch (error) {
    setStatus("camera blocked");
    setValidation(error.message || "camera could not start", "bad");
  }
}

function captureSquare() {
  if (!video.videoWidth || !video.videoHeight) {
    setValidation("camera frame is not ready", "bad");
    return;
  }

  const size = Math.min(video.videoWidth, video.videoHeight);
  const sx = Math.floor((video.videoWidth - size) / 2);
  const sy = Math.floor((video.videoHeight - size) / 2);
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);
  capturedDataUrl = canvas.toDataURL("image/jpeg", 0.94);
  capturePanel.classList.add("has-capture");
  analyzeButton.disabled = false;
  resetSolution();
  setStatus("frame captured");
  setValidation("ready to detect");
}

function uploadImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setValidation("choose an image file", "bad");
    return;
  }

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      capturedDataUrl = String(reader.result);
      await drawDataUrlToCanvas(capturedDataUrl, "contain");
      analyzeButton.disabled = false;
      resetSolution();
      setStatus("image loaded");
      setValidation("ready to detect");
    } catch (error) {
      setStatus("upload failed");
      setValidation(error.message, "bad");
    }
  };
  reader.onerror = () => {
    setStatus("upload failed");
    setValidation("image could not be read", "bad");
  };
  reader.readAsDataURL(file);
}

function confidenceClass(value) {
  if (value >= 0.55) return "good";
  if (value >= 0.28) return "medium";
  return "low";
}

function makeOptions(select, selected) {
  for (const label of window.PUZZLE_LABELS) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    option.selected = label === selected;
    select.append(option);
  }
}

function cropPosition(index) {
  const row = Math.floor(index / 4);
  const col = index % 4;
  return `${col * 33.3333}% ${row * 33.3333}%`;
}

function renderDetected(cells) {
  detectedGrid.innerHTML = "";
  detectedLabels = cells.map((cell) => cell.label);

  for (const cell of cells) {
    const node = cellTemplate.content.firstElementChild.cloneNode(true);
    node.classList.add(confidenceClass(cell.confidence));
    if (cell.label === "00") {
      node.classList.add("blank-cell");
    }
    node.title = `blank ${Math.round((cell.blankScore || 0) * 100)}%, colour ${Math.round((cell.colourCoverage || 0) * 100)}%`;

    const crop = node.querySelector(".tile-crop");
    crop.style.backgroundImage = `url(${capturedDataUrl})`;
    crop.style.backgroundPosition = cropPosition(cell.index);

    const select = node.querySelector("select");
    makeOptions(select, cell.label);
    select.addEventListener("change", () => {
      detectedLabels[cell.index] = select.value;
      updateBoardValidation();
      solveButton.disabled = !canSolve();
    });

    const confidence = node.querySelector(".confidence");
    confidence.textContent = `${Math.round(cell.confidence * 100)}%`;
    detectedGrid.append(node);
  }

  updateBoardValidation();
  solveButton.disabled = !canSolve();
}

function renderPlaceholders() {
  detectedGrid.innerHTML = "";
  detectedLabels = Array(16).fill("00");

  for (let index = 0; index < 16; index += 1) {
    const node = cellTemplate.content.firstElementChild.cloneNode(true);
    const crop = node.querySelector(".tile-crop");
    crop.style.backgroundPosition = cropPosition(index);

    const select = node.querySelector("select");
    makeOptions(select, "00");
    select.disabled = true;

    const confidence = node.querySelector(".confidence");
    confidence.textContent = "--";
    detectedGrid.append(node);
  }
}

function currentProblem() {
  const tileLabels = detectedLabels.filter((label) => label !== "00");
  const missing = window.PUZZLE_LABELS.slice(1).filter((label) => !tileLabels.includes(label));
  const duplicates = [...new Set(tileLabels.filter((label, index) => tileLabels.indexOf(label) !== index))];
  const blankCount = detectedLabels.filter((label) => label === "00").length;
  return { missing, duplicates, blankCount };
}

function inferredOutside() {
  const { missing, blankCount } = currentProblem();
  if (blankCount === 0 && missing.length === 0) return "00";
  if (blankCount === 1 && missing.length === 1) return missing[0];
  return "--";
}

function canSolve() {
  const { missing, duplicates, blankCount } = currentProblem();
  return duplicates.length === 0 && ((blankCount === 0 && missing.length === 0) || (blankCount === 1 && missing.length === 1));
}

function updateBoardValidation() {
  const { missing, duplicates, blankCount } = currentProblem();
  outsideValue.textContent = `outside: ${inferredOutside()}`;

  if (duplicates.length) {
    setValidation(`duplicate: ${duplicates.join(", ")}`, "bad");
    return;
  }

  if (blankCount > 1) {
    setValidation("only one 00 can be in the grid", "bad");
    return;
  }

  if (blankCount === 0 && missing.length > 0) {
    setValidation(`missing: ${missing.join(", ")}`, "bad");
    return;
  }

  if (blankCount === 1 && missing.length !== 1) {
    setValidation("one numbered tile must be outside", "bad");
    return;
  }

  setValidation("board ready", "good");
}

async function analyzeCapture() {
  if (!capturedDataUrl) return;
  setStatus("detecting");
  analyzeButton.disabled = true;
  solveButton.disabled = true;

  try {
    let data;
    if (window.PuzzleCore?.analyzeImage) {
      data = await window.PuzzleCore.analyzeImage(capturedDataUrl, (message) => setValidation(message));
    } else {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: capturedDataUrl }),
      });
      data = await response.json();
      if (!response.ok) throw new Error(data.error || "detection failed");
    }

    if (data.boardImage) {
      capturedDataUrl = data.boardImage;
      await drawDataUrlToCanvas(capturedDataUrl, "cover");
    }
    renderDetected(data.cells);
    outsideValue.textContent = `outside: ${data.outside}`;
    setStatus("detected");
  } catch (error) {
    setStatus("detect failed");
    setValidation(error.message, "bad");
  } finally {
    analyzeButton.disabled = false;
  }
}

function normaliseMoveDetails(moves, moveDetails) {
  if (Array.isArray(moveDetails) && moveDetails.length) {
    return moveDetails;
  }
  return moves.map((move, index) => ({
    step: index + 1,
    tile: move,
    direction: "",
    text: move,
  }));
}

function renderMoves(moves, moveDetails) {
  movesBox.innerHTML = "";
  if (!moves.length) {
    const chip = document.createElement("span");
    chip.className = "move-chip";
    chip.textContent = "solved";
    movesBox.append(chip);
    return;
  }

  normaliseMoveDetails(moves, moveDetails).forEach((move) => {
    const chip = document.createElement("span");
    chip.className = "move-chip";
    chip.dataset.direction = move.direction || "";
    chip.innerHTML = `<span>${move.step}. ${move.tile}</span><strong>${move.direction}</strong>`;
    chip.title = move.from && move.to ? `${move.from} to ${move.to}` : "";
    movesBox.append(chip);
  });
}

function referenceTileUrl(label) {
  return new URL(`reference/${label}.png`, document.baseURI).href;
}

function renderStateBoard(labels) {
  const grid = document.createElement("div");
  grid.className = "board-grid";
  labels.slice(0, 16).forEach((label) => {
    const tile = document.createElement("span");
    tile.className = `board-tile ${label === "00" ? "blank" : ""}`.trim();
    if (label !== "00") {
      tile.classList.add("has-reference");
      tile.style.backgroundImage = `url("${referenceTileUrl(label)}")`;
    }

    const badge = document.createElement("span");
    badge.className = "board-tile-label";
    badge.textContent = label;
    tile.append(badge);
    grid.append(tile);
  });
  return grid;
}

function renderSteps(states, moves, moveDetails) {
  stepsBox.innerHTML = "";
  const details = normaliseMoveDetails(moves, moveDetails);
  states.forEach((state, index) => {
    const card = document.createElement("article");
    card.className = "step-card";
    const title = document.createElement("p");
    title.className = "step-title";
    title.textContent = index === 0 ? "start" : `move ${index}: ${details[index - 1].text}`;
    card.append(title);
    card.append(renderStateBoard(state));
    stepsBox.append(card);
  });
}

async function solveBoard() {
  if (!canSolve()) {
    updateBoardValidation();
    return;
  }

  activeSolveJobId = null;
  setStatus("solving");
  solveButton.disabled = true;
  algorithmSelect.disabled = true;
  movesBox.innerHTML = "";
  stepsBox.innerHTML = "";
  solveStats.textContent = "starting";
  setSolveProgress({ progress: 5, running: true, text: "Queued" });

  try {
    if (window.PuzzleCore?.solve) {
      const result = await window.PuzzleCore.solve(detectedLabels, algorithmSelect.value, ({ progress, text }) => {
        setSolveProgress({ progress, running: true, text });
        solveStats.textContent = "solving";
      });
      setSolveProgress({ progress: 100, running: false, text: `Solved in ${result.stats.seconds}s` });
      renderMoves(result.moves, result.moveDetails);
      renderSteps(result.states, result.moves, result.moveDetails);
      solveStats.textContent = `${result.moves.length} moves - ${result.stats.algorithm} - ${result.stats.seconds}s`;
      outsideValue.textContent = `outside: ${result.outside}`;
      setStatus("solved");
      setValidation("solution ready", "good");
      return;
    }

    const response = await fetch("/api/solve/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels: detectedLabels, algorithm: algorithmSelect.value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "solve failed to start");

    const jobId = data.jobId;
    activeSolveJobId = jobId;
    let result = null;

    while (activeSolveJobId === jobId && !result) {
      await new Promise((resolve) => setTimeout(resolve, 650));
      const statusResponse = await fetch(`/api/solve/status/${encodeURIComponent(jobId)}`, { cache: "no-store" });
      const statusData = await statusResponse.json();
      if (!statusResponse.ok) throw new Error(statusData.error || "solve status failed");

      const elapsed = Number(statusData.elapsed || 0).toFixed(1);
      if (statusData.status === "done") {
        result = statusData.result;
        setSolveProgress({ progress: 100, running: false, text: `Solved in ${elapsed}s` });
      } else if (statusData.status === "error") {
        throw new Error(statusData.message || "solve failed");
      } else {
        const message = statusData.message || "Searching for a solution";
        const progress = Number(statusData.progress || 35);
        setSolveProgress({ progress, running: true, text: `${message} - ${elapsed}s` });
        solveStats.textContent = `solving - ${elapsed}s`;
      }
    }

    if (!result) return;
    renderMoves(result.moves, result.moveDetails);
    renderSteps(result.states, result.moves, result.moveDetails);
    solveStats.textContent = `${result.moves.length} moves - ${result.stats.algorithm} - ${result.stats.seconds}s`;
    outsideValue.textContent = `outside: ${result.outside}`;
    setStatus("solved");
    setValidation("solution ready", "good");
  } catch (error) {
    solveStats.textContent = "blocked";
    setStatus("solve failed");
    setValidation(error.message, "bad");
    setSolveProgress({ progress: 100, running: false, text: error.message });
  } finally {
    activeSolveJobId = null;
    algorithmSelect.disabled = false;
    solveButton.disabled = !canSolve();
  }
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  try {
    await navigator.serviceWorker.register(new URL("sw.js", document.baseURI));
  } catch (error) {
    try {
      await navigator.serviceWorker.register(new URL("static/sw.js", document.baseURI));
    } catch (fallbackError) {
      console.warn("service worker registration failed", error, fallbackError);
    }
  }
}

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  deferredInstallPrompt = event;
  installButton.hidden = false;
});

installButton.addEventListener("click", async () => {
  if (!deferredInstallPrompt) return;
  installButton.hidden = true;
  deferredInstallPrompt.prompt();
  await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
});

window.addEventListener("appinstalled", () => {
  deferredInstallPrompt = null;
  installButton.hidden = true;
});

startCameraButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", captureSquare);
uploadButton.addEventListener("click", () => imageUpload.click());
imageUpload.addEventListener("change", () => uploadImageFile(imageUpload.files[0]));
analyzeButton.addEventListener("click", analyzeCapture);
solveButton.addEventListener("click", solveBoard);

renderPlaceholders();
registerServiceWorker();
startCamera();
