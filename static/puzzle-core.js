(() => {
  const ROWS = 4;
  const COLS = 4;
  const POCKET = ROWS * COLS;
  const BLANK = 0;
  const LABELS = Array.from({ length: 16 }, (_, index) => `${Math.floor(index / 4) + 1}${(index % 4) + 1}`);
  const TILE_LABELS = ["00", ...LABELS];
  const LABEL_TO_TILE = new Map(TILE_LABELS.map((label, index) => [label, index]));
  const GOAL = [...Array.from({ length: 16 }, (_, index) => index + 1), BLANK];
  const GOAL_POS = Array(17).fill(0);
  const GOAL_ROW = Array(17).fill(0);
  const GOAL_COL = Array(17).fill(0);

  for (let tile = 1; tile <= 16; tile += 1) {
    const pos = tile - 1;
    GOAL_POS[tile] = pos;
    GOAL_ROW[tile] = Math.floor(pos / COLS);
    GOAL_COL[tile] = pos % COLS;
  }
  GOAL_POS[BLANK] = POCKET;

  function buildAdjacency() {
    const adjacency = Array.from({ length: POCKET + 1 }, () => []);
    for (let row = 0; row < ROWS; row += 1) {
      for (let col = 0; col < COLS; col += 1) {
        const pos = row * COLS + col;
        for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const nr = row + dr;
          const nc = col + dc;
          if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
            adjacency[pos].push(nr * COLS + nc);
          }
        }
      }
    }
    adjacency[POCKET].push(POCKET - 1);
    adjacency[POCKET - 1].push(POCKET);
    return adjacency;
  }

  const ADJACENCY = buildAdjacency();

  function buildDistances() {
    return ADJACENCY.map((_, start) => {
      const dist = Array(POCKET + 1).fill(1e9);
      const queue = [start];
      dist[start] = 0;
      for (let head = 0; head < queue.length; head += 1) {
        const here = queue[head];
        for (const next of ADJACENCY[here]) {
          if (dist[next] === 1e9) {
            dist[next] = dist[here] + 1;
            queue.push(next);
          }
        }
      }
      return dist;
    });
  }

  const DISTANCES = buildDistances();
  const COLOURS = Array.from({ length: POCKET + 1 }, (_, pos) => {
    if (pos === POCKET) return 1 - ((ROWS - 1 + COLS - 1) % 2);
    const row = Math.floor(pos / COLS);
    const col = pos % COLS;
    return (row + col) % 2;
  });

  function keyOf(state) {
    return state.join(",");
  }

  function normaliseToken(token) {
    const value = String(token).trim();
    return value === "0" ? "00" : value;
  }

  function stateFromGridLabels(labels) {
    if (!Array.isArray(labels) || labels.length !== 16) {
      throw new Error("Expected 16 grid labels.");
    }

    const normalised = labels.map(normaliseToken);
    const invalid = normalised.filter((label) => label !== "00" && !LABEL_TO_TILE.has(label));
    if (invalid.length) throw new Error(`Unknown label(s): ${invalid.join(", ")}`);

    const blankCount = normalised.filter((label) => label === "00").length;
    if (blankCount > 1) throw new Error("Only one grid position can be 00.");

    const tileLabels = normalised.filter((label) => label !== "00");
    const duplicates = [...new Set(tileLabels.filter((label, index) => tileLabels.indexOf(label) !== index))];
    if (duplicates.length) throw new Error(`Duplicate tile(s): ${duplicates.join(", ")}`);

    const missing = LABELS.filter((label) => !tileLabels.includes(label));
    let outside = "00";
    if (blankCount === 0) {
      if (missing.length) throw new Error(`Missing tile(s): ${missing.join(", ")}`);
    } else {
      if (missing.length !== 1) throw new Error("When 00 is in the grid, exactly one numbered tile must be outside.");
      outside = missing[0];
    }

    return {
      state: [...normalised, outside].map((label) => LABEL_TO_TILE.get(label)),
      outside,
    };
  }

  function permutationParity(state) {
    const order = state.map((tile) => GOAL_POS[tile]);
    let inversions = 0;
    for (let i = 0; i < order.length; i += 1) {
      for (let j = i + 1; j < order.length; j += 1) {
        if (order[i] > order[j]) inversions ^= 1;
      }
    }
    return inversions;
  }

  function isSolvable(state) {
    const blankPos = state.indexOf(BLANK);
    return permutationParity(state) === (COLOURS[blankPos] ^ COLOURS[POCKET]);
  }

  function countInversions(values) {
    let total = 0;
    for (let i = 0; i < values.length; i += 1) {
      for (let j = i + 1; j < values.length; j += 1) {
        if (values[i] > values[j]) total += 1;
      }
    }
    return total;
  }

  function distanceHeuristic(state) {
    let total = 0;
    for (let pos = 0; pos < state.length; pos += 1) {
      const tile = state[pos];
      if (tile !== BLANK) total += DISTANCES[pos][GOAL_POS[tile]];
    }
    return total;
  }

  function linearConflict(state) {
    let conflicts = 0;
    for (let row = 0; row < ROWS; row += 1) {
      const goalCols = [];
      for (let col = 0; col < COLS; col += 1) {
        const tile = state[row * COLS + col];
        if (tile !== BLANK && GOAL_ROW[tile] === row) goalCols.push(GOAL_COL[tile]);
      }
      conflicts += countInversions(goalCols);
    }

    for (let col = 0; col < COLS; col += 1) {
      const goalRows = [];
      for (let row = 0; row < ROWS; row += 1) {
        const tile = state[row * COLS + col];
        if (tile !== BLANK && GOAL_COL[tile] === col) goalRows.push(GOAL_ROW[tile]);
      }
      conflicts += countInversions(goalRows);
    }
    return conflicts * 2;
  }

  function heuristic(state) {
    return distanceHeuristic(state) + linearConflict(state);
  }

  class MinHeap {
    constructor() {
      this.items = [];
    }

    get size() {
      return this.items.length;
    }

    push(item) {
      this.items.push(item);
      this.bubbleUp(this.items.length - 1);
    }

    pop() {
      if (this.items.length === 1) return this.items.pop();
      const top = this.items[0];
      this.items[0] = this.items.pop();
      this.bubbleDown(0);
      return top;
    }

    less(left, right) {
      return left[0] < right[0] || (left[0] === right[0] && left[1] < right[1]);
    }

    bubbleUp(index) {
      while (index > 0) {
        const parent = Math.floor((index - 1) / 2);
        if (!this.less(this.items[index], this.items[parent])) break;
        [this.items[index], this.items[parent]] = [this.items[parent], this.items[index]];
        index = parent;
      }
    }

    bubbleDown(index) {
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        let best = index;
        if (left < this.items.length && this.less(this.items[left], this.items[best])) best = left;
        if (right < this.items.length && this.less(this.items[right], this.items[best])) best = right;
        if (best === index) break;
        [this.items[index], this.items[best]] = [this.items[best], this.items[index]];
        index = best;
      }
    }
  }

  function orderedMoves(state, blank) {
    const moves = [];
    for (const nextBlank of ADJACENCY[blank]) {
      const tile = state[nextBlank];
      const oldDistance = DISTANCES[nextBlank][GOAL_POS[tile]];
      const newDistance = DISTANCES[blank][GOAL_POS[tile]];
      const nextState = state.slice();
      nextState[blank] = tile;
      nextState[nextBlank] = BLANK;
      moves.push({
        delta: newDistance - oldDistance,
        nextBlank,
        tile,
        nextState,
        nextH: heuristic(nextState),
      });
    }
    moves.sort((a, b) => a.nextH - b.nextH || a.delta - b.delta);
    return moves;
  }

  function reconstruct(parent, endKey) {
    const moves = [];
    let here = endKey;
    while (parent.has(here)) {
      const entry = parent.get(here);
      moves.push(entry.tile);
      here = entry.previous;
    }
    moves.reverse();
    return moves;
  }

  function pause() {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  async function weightedAStar(state, options = {}) {
    const {
      weight = 3,
      maxNodes = 1500000,
      maxSeconds = 30,
      algorithmName = "Fast weighted A*",
      onProgress = () => {},
    } = options;

    if (!isSolvable(state)) {
      throw new Error("This puzzle is valid but impossible to solve.");
    }
    if (keyOf(state) === keyOf(GOAL)) {
      return { moves: [], stats: { nodes: 0, iterations: 1, seconds: 0, algorithm: algorithmName } };
    }

    const started = performance.now();
    const startH = heuristic(state);
    const heap = new MinHeap();
    const startKey = keyOf(state);
    let tie = 0;
    heap.push([startH * weight, startH, 0, tie, state, state.indexOf(BLANK), startKey]);

    const bestG = new Map([[startKey, 0]]);
    const parent = new Map();
    const closed = new Set();
    let nodes = 0;

    while (heap.size) {
      const elapsed = (performance.now() - started) / 1000;
      if (maxSeconds && elapsed > maxSeconds) {
        throw new Error(`Fast search stopped after ${maxSeconds}s. Check detected labels or try Auto + exact fallback.`);
      }

      const [, currentH, currentG, , current, blank, currentKey] = heap.pop();
      if (closed.has(currentKey)) continue;
      if (currentG !== bestG.get(currentKey)) continue;
      closed.add(currentKey);
      nodes += 1;

      if (nodes > maxNodes) {
        throw new Error(`Fast search stopped after ${maxNodes.toLocaleString()} states. Check detected labels.`);
      }

      if (currentH === 0) {
        return {
          moves: reconstruct(parent, currentKey),
          stats: {
            nodes,
            iterations: 1,
            seconds: Number(((performance.now() - started) / 1000).toFixed(3)),
            algorithm: algorithmName,
          },
        };
      }

      for (const move of orderedMoves(current, blank)) {
        const nextKey = keyOf(move.nextState);
        if (closed.has(nextKey)) continue;
        const nextG = currentG + 1;
        if (nextG < (bestG.get(nextKey) ?? 1e9)) {
          bestG.set(nextKey, nextG);
          parent.set(nextKey, { previous: currentKey, tile: move.tile });
          tie += 1;
          heap.push([nextG + weight * move.nextH, move.nextH, nextG, tie, move.nextState, move.nextBlank, nextKey]);
        }
      }

      if (nodes % 2500 === 0) {
        onProgress({
          progress: 35,
          text: `Searching (${startH}+ moves minimum, ${nodes.toLocaleString()} states)`,
        });
        await pause();
      }
    }

    throw new Error("No solution found.");
  }

  function applyMoves(state, moves) {
    const board = state.slice();
    const states = [board.slice()];
    for (const tile of moves) {
      const blank = board.indexOf(BLANK);
      const tilePos = board.indexOf(tile);
      if (!ADJACENCY[blank].includes(tilePos)) throw new Error(`Move ${TILE_LABELS[tile]} is not legal.`);
      board[blank] = tile;
      board[tilePos] = BLANK;
      states.push(board.slice());
    }
    return states;
  }

  function formatPosition(pos) {
    if (pos === POCKET) return "outside slot";
    return `row ${Math.floor(pos / COLS) + 1}, col ${(pos % COLS) + 1}`;
  }

  function moveDirection(tilePos, blankPos) {
    if (tilePos === POCKET - 1 && blankPos === POCKET) return "down";
    if (tilePos === POCKET && blankPos === POCKET - 1) return "up";
    const tileRow = Math.floor(tilePos / COLS);
    const tileCol = tilePos % COLS;
    const blankRow = Math.floor(blankPos / COLS);
    const blankCol = blankPos % COLS;
    if (blankRow - tileRow === -1 && blankCol === tileCol) return "up";
    if (blankRow - tileRow === 1 && blankCol === tileCol) return "down";
    if (blankCol - tileCol === -1 && blankRow === tileRow) return "left";
    if (blankCol - tileCol === 1 && blankRow === tileRow) return "right";
    throw new Error("No direction for move.");
  }

  function describeMoves(state, moves) {
    const board = state.slice();
    return moves.map((tile, index) => {
      const blank = board.indexOf(BLANK);
      const tilePos = board.indexOf(tile);
      const direction = moveDirection(tilePos, blank);
      board[blank] = tile;
      board[tilePos] = BLANK;
      return {
        step: index + 1,
        tile: TILE_LABELS[tile],
        direction,
        from: formatPosition(tilePos),
        to: formatPosition(blank),
        text: `${TILE_LABELS[tile]} ${direction}`,
      };
    });
  }

  async function solve(labels, algorithm = "fast", onProgress = () => {}) {
    const { state, outside } = stateFromGridLabels(labels);
    const lowerBound = heuristic(state);
    onProgress({ progress: 20, text: `Searching (${lowerBound}+ moves minimum)` });
    const algorithmId = String(algorithm || "fast").toLowerCase();
    let search;
    if (algorithmId === "a-star-closed" || algorithmId === "ida-star") {
      search = await weightedAStar(state, {
        weight: 1,
        maxNodes: 1000000,
        maxSeconds: 45,
        algorithmName: "A* exact",
        onProgress,
      });
    } else {
      search = await weightedAStar(state, {
        weight: 3,
        maxNodes: 1500000,
        maxSeconds: algorithmId === "auto" ? 45 : 30,
        algorithmName: "Fast weighted A*",
        onProgress,
      });
    }

    const states = applyMoves(state, search.moves).map((stepState) => stepState.map((tile) => TILE_LABELS[tile]));
    return {
      outside,
      lowerBound,
      moves: search.moves.map((tile) => TILE_LABELS[tile]),
      moveDetails: describeMoves(state, search.moves),
      states,
      stats: search.stats,
    };
  }

  function assetUrl(path) {
    return new URL(path, document.baseURI).href;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Could not load image: ${src}`));
      image.src = src;
    });
  }

  function canvasFromImage(image) {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    canvas.getContext("2d").drawImage(image, 0, 0);
    return canvas;
  }

  function smoothProjection(values, windowSize) {
    const size = Math.max(3, windowSize % 2 === 0 ? windowSize + 1 : windowSize);
    const radius = Math.floor(size / 2);
    return values.map((_, index) => {
      let total = 0;
      let count = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const pos = index + offset;
        if (pos >= 0 && pos < values.length) {
          total += values[pos];
          count += 1;
        }
      }
      return total / count;
    });
  }

  function longestSegment(active, minimum) {
    let best = null;
    let start = null;
    for (let index = 0; index < active.length; index += 1) {
      if (active[index] && start === null) start = index;
      if ((!active[index] || index === active.length - 1) && start !== null) {
        const end = active[index] ? index + 1 : index;
        if (end - start >= minimum && (!best || end - start > best[1] - best[0])) best = [start, end];
        start = null;
      }
    }
    return best;
  }

  function expandToSquare(box, width, height) {
    let [left, top, right, bottom] = box;
    const side = Math.max(right - left, bottom - top);
    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    left = Math.round(cx - side / 2);
    top = Math.round(cy - side / 2);
    right = left + side;
    bottom = top + side;
    if (left < 0) {
      right -= left;
      left = 0;
    }
    if (top < 0) {
      bottom -= top;
      top = 0;
    }
    if (right > width) {
      left -= right - width;
      right = width;
    }
    if (bottom > height) {
      top -= bottom - height;
      bottom = height;
    }
    return [Math.max(0, left), Math.max(0, top), Math.min(width, right), Math.min(height, bottom)];
  }

  function detectBoardBox(sourceCanvas) {
    const longestSide = Math.max(sourceCanvas.width, sourceCanvas.height);
    const scale = longestSide > 900 ? 900 / longestSide : 1;
    const work = document.createElement("canvas");
    work.width = Math.round(sourceCanvas.width * scale);
    work.height = Math.round(sourceCanvas.height * scale);
    const ctx = work.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(sourceCanvas, 0, 0, work.width, work.height);
    const data = ctx.getImageData(0, 0, work.width, work.height).data;
    const rowProjection = Array(work.height).fill(0);
    const colProjection = Array(work.width).fill(0);
    let activeCount = 0;

    for (let y = 0; y < work.height; y += 1) {
      for (let x = 0; x < work.width; x += 1) {
        const offset = (y * work.width + x) * 4;
        const r = data[offset] / 255;
        const g = data[offset + 1] / 255;
        const b = data[offset + 2] / 255;
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        const saturation = (mx - mn) / (mx + 1e-6);
        if (saturation > 0.22 && mx > 0.2) {
          rowProjection[y] += 1;
          colProjection[x] += 1;
          activeCount += 1;
        }
      }
    }

    if (activeCount / (work.width * work.height) < 0.02) return null;

    const rows = smoothProjection(rowProjection.map((value) => value / work.width), Math.max(5, Math.floor(work.height / 80)));
    const rowThreshold = Math.max(0.12, Math.max(...rows) * 0.36);
    const ySegment = longestSegment(rows.map((value) => value > rowThreshold), Math.floor(work.height * 0.28));
    if (!ySegment) return null;

    const cols = smoothProjection(colProjection.map((value) => value / Math.max(1, ySegment[1] - ySegment[0])), Math.max(5, Math.floor(work.width / 80)));
    const colThreshold = Math.max(0.1, Math.max(...cols) * 0.34);
    const xSegment = longestSegment(cols.map((value) => value > colThreshold), Math.floor(work.width * 0.28));
    if (!xSegment) return null;

    let [x1, x2] = xSegment;
    let [y1, y2] = ySegment;
    const padX = Math.floor((x2 - x1) * 0.012);
    const padY = Math.floor((y2 - y1) * 0.012);
    x1 = Math.max(0, x1 - padX);
    x2 = Math.min(work.width, x2 + padX);
    y1 = Math.max(0, y1 - padY);
    y2 = Math.min(work.height, y2 + padY);

    const box = expandToSquare([x1 / scale, y1 / scale, x2 / scale, y2 / scale], sourceCanvas.width, sourceCanvas.height);
    if (Math.min(box[2] - box[0], box[3] - box[1]) < Math.min(sourceCanvas.width, sourceCanvas.height) * 0.25) return null;
    return box;
  }

  function cropBoard(sourceCanvas) {
    const box = detectBoardBox(sourceCanvas);
    let cropBox = box;
    if (!cropBox) {
      const side = Math.min(sourceCanvas.width, sourceCanvas.height);
      const left = (sourceCanvas.width - side) / 2;
      const top = (sourceCanvas.height - side) / 2;
      cropBox = [left, top, left + side, top + side];
    }
    const side = Math.min(cropBox[2] - cropBox[0], cropBox[3] - cropBox[1]);
    const board = document.createElement("canvas");
    board.width = 960;
    board.height = 960;
    board.getContext("2d").drawImage(sourceCanvas, cropBox[0], cropBox[1], side, side, 0, 0, 960, 960);
    return { board, cropBox: box };
  }

  function drawCellToCanvas(board, index, marginRatio, size) {
    const row = Math.floor(index / 4);
    const col = index % 4;
    const cell = board.width / 4;
    const margin = cell * marginRatio;
    const srcX = col * cell + margin;
    const srcY = row * cell + margin;
    const srcSize = cell - margin * 2;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    canvas.getContext("2d").drawImage(board, srcX, srcY, srcSize, srcSize, 0, 0, size, size);
    return canvas;
  }

  function normalise(values) {
    let total = 0;
    for (const value of values) total += value;
    const mean = total / values.length;
    let variance = 0;
    for (const value of values) variance += (value - mean) ** 2;
    const std = Math.sqrt(variance / values.length) + 1e-6;
    return Float32Array.from(values, (value) => (value - mean) / std);
  }

  function featuresFromCanvas(canvas, size = 64) {
    const work = document.createElement("canvas");
    work.width = size;
    work.height = size;
    const ctx = work.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    const grayRaw = new Float32Array(size * size);
    const hist = new Float32Array(48);

    for (let i = 0; i < size * size; i += 1) {
      const offset = i * 4;
      const r = data[offset] / 255;
      const g = data[offset + 1] / 255;
      const b = data[offset + 2] / 255;
      grayRaw[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      hist[Math.min(15, Math.floor(r * 16))] += 1;
      hist[16 + Math.min(15, Math.floor(g * 16))] += 1;
      hist[32 + Math.min(15, Math.floor(b * 16))] += 1;
    }

    for (let i = 0; i < hist.length; i += 1) hist[i] /= size * size;
    const gray = normalise(grayRaw);
    const edgeRaw = new Float32Array(size * size);
    for (let y = 1; y < size - 1; y += 1) {
      for (let x = 1; x < size - 1; x += 1) {
        const i = y * size + x;
        const gx = grayRaw[i + 1] - grayRaw[i - 1];
        const gy = grayRaw[i + size] - grayRaw[i - size];
        edgeRaw[i] = Math.sqrt(gx * gx + gy * gy);
      }
    }
    return { gray, edge: normalise(edgeRaw), hist };
  }

  function metricsFromCanvas(canvas, size = 64) {
    const work = document.createElement("canvas");
    work.width = size;
    work.height = size;
    const ctx = work.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    const gray = new Float32Array(size * size);
    let colour = 0;
    let white = 0;
    let dark = 0;
    let brightness = 0;
    let saturationSum = 0;

    for (let i = 0; i < size * size; i += 1) {
      const offset = i * 4;
      const r = data[offset] / 255;
      const g = data[offset + 1] / 255;
      const b = data[offset + 2] / 255;
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      const saturation = (mx - mn) / (mx + 1e-6);
      gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      if (saturation > 0.2 && mx > 0.24) colour += 1;
      if (saturation < 0.16 && mx > 0.58) white += 1;
      if (mx < 0.3) dark += 1;
      brightness += mx;
      saturationSum += saturation;
    }

    let texture = 0;
    for (let y = 1; y < size - 1; y += 1) {
      for (let x = 1; x < size - 1; x += 1) {
        const i = y * size + x;
        const gx = gray[i + 1] - gray[i - 1];
        const gy = gray[i + size] - gray[i - size];
        texture += Math.sqrt(gx * gx + gy * gy);
      }
    }
    texture /= size * size;

    const count = size * size;
    const colourCoverage = colour / count;
    const whiteCoverage = white / count;
    const darkCoverage = dark / count;
    const meanBrightness = brightness / count;
    const meanSaturation = saturationSum / count;
    const blankScore = Math.max(0, Math.min(1,
      0.46 * whiteCoverage +
      0.34 * (1 - colourCoverage) +
      0.13 * (1 - meanSaturation) +
      0.07 * meanBrightness -
      0.18 * darkCoverage -
      0.2 * texture,
    ));

    return { colourCoverage, whiteCoverage, darkCoverage, meanBrightness, meanSaturation, texture, blankScore };
  }

  function correlationCost(left, right) {
    let dot = 0;
    let leftNorm = 0;
    let rightNorm = 0;
    for (let i = 0; i < left.length; i += 1) {
      dot += left[i] * right[i];
      leftNorm += left[i] * left[i];
      rightNorm += right[i] * right[i];
    }
    const corr = dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm) + 1e-6);
    return Math.max(0, Math.min(1, (1 - corr) / 2));
  }

  function featureCost(left, right) {
    let hist = 0;
    for (let i = 0; i < left.hist.length; i += 1) hist += Math.abs(left.hist[i] - right.hist[i]);
    hist /= 6;
    return 0.58 * correlationCost(left.gray, right.gray) + 0.27 * correlationCost(left.edge, right.edge) + 0.15 * hist;
  }

  let referenceCache = null;
  let trainedModelCache = undefined;

  async function referenceFeatures() {
    if (referenceCache) return referenceCache;
    const refs = new Map();
    await Promise.all(LABELS.map(async (label) => {
      const image = await loadImage(assetUrl(`reference/${label}.png`));
      refs.set(label, featuresFromCanvas(canvasFromImage(image)));
    }));
    referenceCache = refs;
    return refs;
  }

  function asFeatureVectors(entry) {
    return {
      gray: Float32Array.from(entry.gray || []),
      edge: Float32Array.from(entry.edge || []),
      hist: Float32Array.from(entry.hist || []),
    };
  }

  async function trainedModel() {
    if (trainedModelCache !== undefined) return trainedModelCache;
    try {
      const response = await fetch(assetUrl("static/trained-model.json"), { cache: "no-store" });
      if (!response.ok) {
        trainedModelCache = null;
        return trainedModelCache;
      }
      const raw = await response.json();
      const labels = new Map();
      for (const [label, prototypes] of Object.entries(raw.prototypes || {})) {
        labels.set(label, prototypes.map((prototype) => ({
          count: Number(prototype.count || 0),
          feature: asFeatureVectors(prototype),
        })));
      }
      trainedModelCache = {
        version: raw.version || 1,
        labels,
        sampleCount: Number(raw.sampleCount || 0),
      };
    } catch (_error) {
      trainedModelCache = null;
    }
    return trainedModelCache;
  }

  function trainedCostForLabel(features, model, label) {
    if (!model?.labels?.has(label)) return null;
    const prototypes = model.labels.get(label);
    if (!prototypes.length) return null;
    let best = Infinity;
    for (const prototype of prototypes) {
      for (const feature of features) {
        best = Math.min(best, featureCost(feature, prototype.feature));
      }
    }
    return best;
  }

  function bestCandidatesFromVariants(features, refs, model) {
    return LABELS.map((label) => {
      const ref = refs.get(label);
      const referenceCost = Math.min(...features.map((feature) => featureCost(feature, ref)));
      const trainedCost = trainedCostForLabel(features, model, label);
      const cost = trainedCost === null ? referenceCost : Math.min(referenceCost, trainedCost * 0.86);
      return { label, cost };
    }).sort((a, b) => a.cost - b.cost);
  }

  function solveAssignment(costRows) {
    const full = 1 << 16;
    let dp = new Float64Array(full);
    let paths = new Array(full);
    dp.fill(Infinity);
    dp[0] = 0;
    paths[0] = [];

    for (const row of costRows) {
      const next = new Float64Array(full);
      const nextPaths = new Array(full);
      next.fill(Infinity);
      for (let mask = 0; mask < full; mask += 1) {
        if (!Number.isFinite(dp[mask])) continue;
        for (let tile = 1; tile <= 16; tile += 1) {
          const bit = 1 << (tile - 1);
          if (mask & bit) continue;
          const newMask = mask | bit;
          const value = dp[mask] + row[tile];
          if (value < next[newMask]) {
            next[newMask] = value;
            nextPaths[newMask] = [...paths[mask], tile];
          }
        }
      }
      dp = next;
      paths = nextPaths;
    }
    let bestMask = -1;
    let bestCost = Infinity;
    for (let mask = 0; mask < full; mask += 1) {
      if (Number.isFinite(dp[mask]) && dp[mask] < bestCost) {
        bestMask = mask;
        bestCost = dp[mask];
      }
    }
    if (bestMask < 0 || !paths[bestMask]) {
      throw new Error("Could not assign reference tiles to the detected board.");
    }
    return { cost: bestCost, assignment: paths[bestMask] };
  }

  function median(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function detectBlankIndex(metrics, ranked, assignedCosts, blankCosts = null) {
    const medianCost = median(assignedCosts);
    if (blankCosts?.length) {
      let bestBlankIndex = 0;
      for (let index = 1; index < blankCosts.length; index += 1) {
        if (blankCosts[index] < blankCosts[bestBlankIndex]) bestBlankIndex = index;
      }
      const bestBlankCost = blankCosts[bestBlankIndex];
      if (
        Number.isFinite(bestBlankCost) &&
        bestBlankCost < 0.22 &&
        assignedCosts[bestBlankIndex] - bestBlankCost > 0.025
      ) {
        return bestBlankIndex;
      }
    }

    const ordered = metrics.map((_, index) => index).sort((a, b) => metrics[b].blankScore - metrics[a].blankScore);
    const bestIndex = ordered[0];
    const bestMetrics = metrics[bestIndex];
    const secondScore = metrics[ordered[1]]?.blankScore ?? 0;
    const bestTileCost = assignedCosts[bestIndex];

    if (
      bestMetrics.blankScore >= 0.66 &&
      bestMetrics.colourCoverage <= 0.3 &&
      bestMetrics.whiteCoverage >= 0.48 &&
      bestMetrics.blankScore - secondScore >= 0.08
    ) {
      return bestIndex;
    }

    if (
      bestMetrics.blankScore >= 0.56 &&
      bestMetrics.colourCoverage <= 0.42 &&
      bestTileCost > Math.max(0.14, medianCost * 1.25)
    ) {
      return bestIndex;
    }

    const worstIndex = assignedCosts.indexOf(Math.max(...assignedCosts));
    const worstCost = assignedCosts[worstIndex];
    const worstGap = ranked[worstIndex][1] ? ranked[worstIndex][1].cost - ranked[worstIndex][0].cost : 0;
    if (
      worstCost > Math.max(0.18, medianCost * 1.55) &&
      worstGap < 0.055 &&
      metrics[worstIndex].blankScore > 0.38
    ) {
      return worstIndex;
    }

    return null;
  }

  async function analyzeImage(dataUrl, onProgress = () => {}) {
    onProgress("loading references");
    const [refs, model] = await Promise.all([referenceFeatures(), trainedModel()]);
    const image = await loadImage(dataUrl);
    const source = canvasFromImage(image);
    const { board, cropBox } = cropBoard(source);
    const boardImage = board.toDataURL("image/jpeg", 0.9);

    onProgress("matching tiles");
    const metrics = [];
    const ranked = [];
    const blankCosts = [];
    for (let index = 0; index < 16; index += 1) {
      metrics.push(metricsFromCanvas(drawCellToCanvas(board, index, 0.11, 64)));
      const variants = [0.035, 0.055, 0.08].map((margin) => featuresFromCanvas(drawCellToCanvas(board, index, margin, 64)));
      ranked.push(bestCandidatesFromVariants(variants, refs, model));
      blankCosts.push(trainedCostForLabel(variants, model, "00") ?? Infinity);
      if (index % 4 === 3) await pause();
    }

    const costRows = ranked.map((candidates) => {
      const row = Array(17).fill(0);
      for (const candidate of candidates) row[LABEL_TO_TILE.get(candidate.label)] = candidate.cost;
      return row;
    });

    const assignment = solveAssignment(costRows).assignment;
    const assignedCosts = assignment.map((tile, index) => costRows[index][tile]);
    const blankIndex = detectBlankIndex(metrics, ranked, assignedCosts, blankCosts);

    let labels;
    if (blankIndex !== null) {
      const reducedRows = costRows.filter((_, index) => index !== blankIndex);
      const reducedAssignment = solveAssignment(reducedRows).assignment;
      let cursor = 0;
      labels = Array.from({ length: 16 }, (_, index) => {
        if (index === blankIndex) return "00";
        if (cursor >= reducedAssignment.length) {
          throw new Error("Could not assign all non-empty puzzle tiles.");
        }
        const label = TILE_LABELS[reducedAssignment[cursor]];
        cursor += 1;
        return label;
      });
    } else {
      labels = assignment.map((tile) => TILE_LABELS[tile]);
    }

    const cells = labels.map((label, index) => {
      const candidates = ranked[index].slice(0, 4);
      const best = candidates[0].cost;
      const second = candidates[1]?.cost ?? best;
      const confidence = label === "00"
        ? metrics[index].blankScore
        : Math.max(0, Math.min(1, (second - best) / 0.16));
      return {
        index,
        row: Math.floor(index / 4) + 1,
        col: (index % 4) + 1,
        label,
        confidence: Number(confidence.toFixed(3)),
        blankScore: Number(metrics[index].blankScore.toFixed(3)),
        colourCoverage: Number(metrics[index].colourCoverage.toFixed(3)),
        candidates: candidates.map((candidate) => ({ label: candidate.label, score: Number((1 - candidate.cost).toFixed(3)) })),
        modelBlankScore: Number((Number.isFinite(blankCosts[index]) ? 1 - blankCosts[index] : 0).toFixed(3)),
      };
    });

    const missing = LABELS.filter((label) => !labels.includes(label));
    return {
      cells,
      labels,
      outside: missing[0] || "00",
      blankIndex,
      boardImage,
      cropBox,
      referenceDir: "reference",
      trainedSamples: model?.sampleCount || 0,
    };
  }

  window.PuzzleCore = {
    labels: LABELS,
    allLabels: TILE_LABELS,
    analyzeImage,
    solve,
    stateFromGridLabels,
    heuristic,
  };
})();
