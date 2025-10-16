const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = 10;
ctx.strokeStyle = '#000';

const preview = document.getElementById('preview');
const previewCtx = preview.getContext('2d');

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
  ctx.clearRect(0, 0, stage.width, stage.height);
  previewCtx.clearRect(0, 0, preview.width, preview.height);
});

document.getElementById('capture').addEventListener('click', () => {
  const downscaled = document.createElement('canvas');
  downscaled.width = 28;
  downscaled.height = 28;
  const dctx = downscaled.getContext('2d');
  dctx.drawImage(stage, 0, 0, 28, 28);
  
  previewCtx.clearRect(0, 0, 28, 28);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.drawImage(downscaled, 0, 0, 28, 28);

  const data = dctx.getImageData(0, 0, 28, 28).data;
  const grayscale = [];
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const alpha = data[i + 3] / 255;
    const gray = Math.round((0.299 * r + 0.587 * g + 0.114 * b) * alpha);
    grayscale.push(gray / 255);
  }

  console.log('Grayscale 28×28 (flattened length = 784):', grayscale);
});