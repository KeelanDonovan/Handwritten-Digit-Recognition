from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import predict_digit

app = FastAPI(title="Digit Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the HTML drawing page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Receives drawn digit, runs CNN inference, returns prediction."""
    pred = await predict_digit(image)
    return JSONResponse({"prediction": int(pred)})