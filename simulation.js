const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d", { alpha: false });
ctx.imageSmoothingEnabled = false;

const heatSlider = document.getElementById("heatSlider");
const diffusionSlider = document.getElementById("diffusionSlider");
const scaleSlider = document.getElementById("scaleSlider");
const toggleButton = document.getElementById("toggleButton");

const gridWidth = 160;
const gridHeight = 200;
const gridSize = gridWidth * gridHeight;
canvas.width = gridWidth;
canvas.height = gridHeight;

const temp = new Float32Array(gridSize);
const tempBuffer = new Float32Array(gridSize);
const velX = new Float32Array(gridSize);
const velY = new Float32Array(gridSize);
const velXBuffer = new Float32Array(gridSize);
const velYBuffer = new Float32Array(gridSize);

const imageData = ctx.createImageData(gridWidth, gridHeight);

const ambientTemp = 0.18;
const maxTemp = 1.0;
const dt = 0.9;

let heatPower = sliderToHeat(heatSlider.value);
let diffusionAmount = sliderToDiffusion(diffusionSlider.value);
let animationHandle = null;
let running = true;

initializeFields();
updateCanvasScale(scaleSlider.value);
startAnimation();

toggleButton.addEventListener("click", () => {
  running = !running;
  toggleButton.textContent = running ? "一時停止" : "再開";
  if (running) {
    startAnimation();
  } else if (animationHandle) {
    cancelAnimationFrame(animationHandle);
    animationHandle = null;
  }
});

heatSlider.addEventListener("input", (event) => {
  heatPower = sliderToHeat(event.target.value);
});

diffusionSlider.addEventListener("input", (event) => {
  diffusionAmount = sliderToDiffusion(event.target.value);
});

scaleSlider.addEventListener("input", (event) => {
  updateCanvasScale(event.target.value);
});

function initializeFields() {
  for (let i = 0; i < gridSize; i += 1) {
    temp[i] = ambientTemp + (Math.random() - 0.5) * 0.01;
    velX[i] = 0;
    velY[i] = 0;
  }
}

function startAnimation() {
  const loop = () => {
    stepSimulation();
    render();
    if (running) {
      animationHandle = requestAnimationFrame(loop);
    }
  };
  animationHandle = requestAnimationFrame(loop);
}

function stepSimulation() {
  applyDamping();
  addHeatSource();
  addBuoyancy();
  addCirculation();
  advectVelocity();
  diffuseField(velX, velXBuffer, 0.0008);
  diffuseField(velY, velYBuffer, 0.0008);
  confineVelocities();
  advectTemperature();
  diffuseField(temp, tempBuffer, diffusionAmount);
  relaxBoundaries();
}

function applyDamping() {
  const damping = 0.995;
  for (let i = 0; i < gridSize; i += 1) {
    velX[i] *= damping;
    velY[i] *= damping;
  }
}

function addHeatSource() {
  const centerX = gridWidth - 16;
  const centerY = gridHeight - 20;
  const radius = gridWidth * 0.16;
  for (let y = gridHeight - Math.floor(radius * 1.4); y < gridHeight - 2; y += 1) {
    for (let x = gridWidth - Math.floor(radius * 1.6); x < gridWidth - 2; x += 1) {
      const dx = x - centerX;
      const dy = (y - centerY) * 1.15;
      const distanceSq = dx * dx + dy * dy;
      if (distanceSq < radius * radius) {
        const idx = index(x, y);
        const falloff = 1 - distanceSq / (radius * radius);
        temp[idx] = Math.min(maxTemp, temp[idx] + heatPower * falloff);
        velY[idx] -= heatPower * 6 * falloff;
        velX[idx] -= heatPower * 1.6 * falloff;
      }
    }
  }
}

function addBuoyancy() {
  const buoyancyStrength = 0.06;
  // 冷たい水が沈む力を、温かい水が上がる力と同じ強さに変更
  const sinkStrength = 0.06; 
  for (let y = 1; y < gridHeight - 1; y += 1) {
    for (let x = 1; x < gridWidth - 1; x += 1) {
      const idx = index(x, y);
      const delta = temp[idx] - ambientTemp;
      if (delta > 0) {
        // 温かい水は上がる（Y方向のマイナスが上）
        velY[idx] -= delta * buoyancyStrength;
      } else {
        // 冷たい水は沈む（Y方向のプラスが下）
        velY[idx] -= delta * sinkStrength;
      }
    }
  }
}

function addCirculation() {
  const pivotX = gridWidth * 0.55;
  const pivotY = gridHeight * 0.55;
  const swirlStrength = 0.0009;
  const returnFlow = 0.0035;
  for (let y = 1; y < gridHeight - 1; y += 1) {
    for (let x = 1; x < gridWidth - 1; x += 1) {
      const idx = index(x, y);
      const dx = x - pivotX;
      const dy = y - pivotY;
      const distance = Math.sqrt(dx * dx + dy * dy) + 1;
      const heatFactor = Math.max(0, temp[idx] - ambientTemp);
      const swirl = swirlStrength * (0.4 + heatFactor * 2);
      velX[idx] += (-dy / distance) * swirl;
      velY[idx] += (dx / distance) * swirl;
      const verticalRatio = y / gridHeight;
      velX[idx] += (0.5 - verticalRatio) * returnFlow * (0.4 + heatFactor * 3);
    }
  }
}

function advectVelocity() {
  velXBuffer.set(velX);
  velYBuffer.set(velY);
  advectField(velX, velXBuffer, velXBuffer, velYBuffer);
  advectField(velY, velYBuffer, velXBuffer, velYBuffer);
}

function advectTemperature() {
  tempBuffer.set(temp);
  advectField(temp, tempBuffer, velX, velY);
}

function advectField(field, source, vx, vy) {
  const maxX = gridWidth - 1.001;
  const maxY = gridHeight - 1.001;
  for (let y = 1; y < gridHeight - 1; y += 1) {
    for (let x = 1; x < gridWidth - 1; x += 1) {
      const idx = index(x, y);
      const backX = clamp(x - vx[idx] * dt, 0.001, maxX);
      const backY = clamp(y - vy[idx] * dt, 0.001, maxY);
      field[idx] = bilinearSample(source, backX, backY);
    }
  }
}

function diffuseField(field, buffer, rate) {
  if (rate <= 0) {
    return;
  }
  buffer.set(field);
  for (let y = 1; y < gridHeight - 1; y += 1) {
    for (let x = 1; x < gridWidth - 1; x += 1) {
      const idx = index(x, y);
      const sum =
        buffer[index(x - 1, y)] +
        buffer[index(x + 1, y)] +
        buffer[index(x, y - 1)] +
        buffer[index(x, y + 1)];
      field[idx] = buffer[idx] + rate * (sum - 4 * buffer[idx]);
    }
  }
}

function relaxBoundaries() {
  const blend = 0.08;
  for (let x = 0; x < gridWidth; x += 1) {
    temp[index(x, 0)] = lerp(temp[index(x, 0)], ambientTemp, blend);
    temp[index(x, gridHeight - 1)] = lerp(temp[index(x, gridHeight - 1)], ambientTemp, blend);
    velY[index(x, 0)] = 0;
    velY[index(x, gridHeight - 1)] = 0;
  }
  for (let y = 0; y < gridHeight; y += 1) {
    temp[index(0, y)] = lerp(temp[index(0, y)], ambientTemp, blend);
    temp[index(gridWidth - 1, y)] = lerp(temp[index(gridWidth - 1, y)], ambientTemp, blend);
    velX[index(0, y)] = 0;
    velX[index(gridWidth - 1, y)] = 0;
  }
}

function confineVelocities() {
  const maxSpeed = 2.6;
  for (let i = 0; i < gridSize; i += 1) {
    velX[i] = clamp(velX[i], -maxSpeed, maxSpeed);
    velY[i] = clamp(velY[i], -maxSpeed, maxSpeed);
  }
}

function render() {
  const data = imageData.data;
  for (let i = 0; i < gridSize; i += 1) {
    const color = tempToColor(temp[i]);
    const offset = i * 4;
    data[offset] = color.r;
    data[offset + 1] = color.g;
    data[offset + 2] = color.b;
    data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function tempToColor(value) {
  const cold = { r: 30, g: 90, b: 220 };
  const mid = { r: 130, g: 110, b: 220 };
  const hot = { r: 255, g: 130, b: 210 };
  const t = clamp((value - ambientTemp) / (maxTemp - ambientTemp), 0, 1);
  if (t < 0.5) {
    const local = t * 2;
    return {
      r: Math.round(lerp(cold.r, mid.r, local)),
      g: Math.round(lerp(cold.g, mid.g, local)),
      b: Math.round(lerp(cold.b, mid.b, local)),
    };
  }
  const local = (t - 0.5) * 2;
  return {
    r: Math.round(lerp(mid.r, hot.r, local)),
    g: Math.round(lerp(mid.g, hot.g, local)),
    b: Math.round(lerp(mid.b, hot.b, local)),
  };
}

function sliderToHeat(value) {
  return 0.015 + (Number(value) / 100) * 0.04;
}

function sliderToDiffusion(value) {
  return 0.005 + (Number(value) / 100) * 0.035;
}

function updateCanvasScale(value) {
  const scale = Number(value) / 100;
  const baseWidth = gridWidth * 3;
  const baseHeight = gridHeight * 3;
  canvas.style.width = `${(baseWidth * scale).toFixed(0)}px`;
  canvas.style.height = `${(baseHeight * scale).toFixed(0)}px`;
}

function bilinearSample(field, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, gridWidth - 1);
  const y1 = Math.min(y0 + 1, gridHeight - 1);
  const tx = x - x0;
  const ty = y - y0;
  const idx00 = index(x0, y0);
  const idx10 = index(x1, y0);
  const idx01 = index(x0, y1);
  const idx11 = index(x1, y1);
  const top = lerp(field[idx00], field[idx10], tx);
  const bottom = lerp(field[idx01], field[idx11], tx);
  return lerp(top, bottom, ty);
}

function index(x, y) {
  return y * gridWidth + x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
