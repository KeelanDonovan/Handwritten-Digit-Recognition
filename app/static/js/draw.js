const CANVAS_BG = '#111';
const INK_COLOR = '#fff';

const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
const preview = document.getElementById('preview');
const previewCtx = preview.getContext('2d');
const captureButton = document.getElementById('capture');
const predictionLabel = document.getElementById('prediction');
const defaultPredictionText = predictionLabel.textContent;

ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = 15;
ctx.strokeStyle = INK_COLOR;

function resetStage() {
  ctx.save();
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, stage.width, stage.height);
  ctx.restore();
}

function clearPreview() {
  previewCtx.save();
  previewCtx.fillStyle = CANVAS_BG;
  previewCtx.fillRect(0, 0, preview.width, preview.height);
  previewCtx.restore();
}

resetStage();
clearPreview();

let drawing = false;

function getPos(event) {
  const rect = stage.getBoundingClientRect();
  const touch = event.touches ? event.touches[0] : event;
  return {
    x: (touch.clientX - rect.left) * (stage.width / rect.width),
    y: (touch.clientY - rect.top) * (stage.height / rect.height)
  };
}

function startDraw(event) {
  drawing = true;
  const pos = getPos(event);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const pos = getPos(event);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function endDraw() {
  drawing = false;
  ctx.closePath();
}

stage.addEventListener('mousedown', startDraw);
stage.addEventListener('mousemove', draw);
stage.addEventListener('mouseup', endDraw);

stage.addEventListener('touchstart', startDraw, { passive: false });
stage.addEventListener('touchmove', draw, { passive: false });
stage.addEventListener('touchend', endDraw);

document.getElementById('clear').addEventListener('click', () => {
  resetStage();
  clearPreview();
  predictionLabel.textContent = defaultPredictionText;
});


captureButton.addEventListener('click', () => {
  const downscaled = document.createElement('canvas');
  downscaled.width = 28;
  downscaled.height = 28;
  const dctx = downscaled.getContext('2d');
  dctx.fillStyle = CANVAS_BG;
  dctx.fillRect(0, 0, 28, 28);
  dctx.drawImage(stage, 0, 0, 28, 28);

  clearPreview();
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.drawImage(downscaled, 0, 0, 28, 28);

  predictionLabel.textContent = 'Sending to model…';
  captureButton.disabled = true;
  captureButton.textContent = 'Predicting…';

  downscaled.toBlob(async (blob) => {
    try {
      const formData = new FormData();
      formData.append('image', blob, 'digit.png');

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const { prediction } = await response.json();
      predictionLabel.textContent = `Model prediction: ${prediction}`;
    } catch (error) {
      console.error('Prediction failed', error);
      predictionLabel.textContent = 'Prediction failed. Try again after redrawing.';
    } finally {
      captureButton.disabled = false;
      captureButton.textContent = 'Capture';
    }
  }, 'image/png');
});