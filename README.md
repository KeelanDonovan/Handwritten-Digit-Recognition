# Handwritten Digit Recognition

This project showcases a custom handwritten digit classifier built around a compact PyTorch CNN. The web app exists to give that model an interactive surface: draw a digit in the browser, the FastAPI service preprocesses it the same way the network was trained, and you get the prediction back instantly.

## Overview

- **Frontend**: `app/templates/index.html` with vanilla JavaScript (`app/static/js/draw.js`) handles the canvas, downscales drawings to 28×28, and posts them to the API.
- **Backend**: `app/main.py` exposes two routes: `/` serves the page and `/predict` accepts an image upload. Static assets are mounted from `app/static`.
- **Model**: `app/digit_recog_cnn.py` defines a compact CNN trained on MNIST. Its weights live in `saved_models/digit_recog_cnn.pt` and are loaded by `app/inference.py`.
- **Data & experiments**: MNIST download artifacts live under `data/`, while `model.ipynb` and `notebooks/` capture the training work.

## Directory layout

```
app/
├── main.py               FastAPI application
├── inference.py          Model loading and preprocessing
├── digit_recog_cnn.py    CNN definition
├── static/
│   ├── css/canvas.css    Styling for the dark theme canvas
│   └── js/draw.js        Drawing, downscaling, and upload logic
└── templates/index.html  Canvas page served from FastAPI

data/                     MNIST assets pulled by torchvision
model.ipynb               Notebook used for training/testing
notebooks/                Additional experiment notebooks
saved_models/digit_recog_cnn.pt   Trained weights
requirements.txt          Conda environment export
```

## Getting started

This repository’s `requirements.txt` is a Conda export (note the `package=version=build` format). The easiest path is to create the environment with Conda:

```bash
conda create -n digit-recog --file requirements.txt
conda activate digit-recog
```

If you prefer `pip`, install the essentials manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn torch torchvision pillow numpy
```

With the environment ready, launch the app from the app folder:

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/` in a browser. Draw a digit, click **Capture**, and the prediction appears under the buttons. The preview tile mirrors the 28×28 input passed to the network.

## API reference

| Method | Path       | Description                                       |
|--------|------------|---------------------------------------------------|
| GET    | `/`        | Renders the drawing interface (Jinja template).   |
| POST   | `/predict` | Accepts `image` in multipart form data, returns `{"prediction": int}`. |

You can probe the prediction route directly:

```bash
curl -X POST \
  -F "image=@sample.png" \
  http://127.0.0.1:8000/predict
```

## Model details

- Input preprocessing mirrors MNIST: convert to grayscale, resize to 28×28, convert to tensor, and normalize with mean `0.1307` and std `0.3081`.
- The CNN uses three convolutional layers (channels 8 → 16 → 32) with ReLU and max pooling, followed by a fully connected layer. Output is a log-softmax over ten digits.
- `app/inference.py` loads `saved_models/digit_recog_cnn.pt` at startup. Update both the weights and the architecture file if you retrain with a different layout.
- There is a `print(logits)` call inside `predict_digit` that can be removed once you are satisfied with the outputs.

To retrain, open `model.ipynb`, run the training cells, and export new weights:

```python
torch.save(model.state_dict(), "saved_models/digit_recog_cnn.pt")
```

Restart the FastAPI server so the new weights are picked up.

## Development notes
- MNIST data under `data/` is optional at runtime but handy for notebook experiments.
