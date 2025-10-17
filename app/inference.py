import torch
from torchvision import transforms
from PIL import Image
import io
from digit_recog_cnn import DigitRecogCNN

model = DigitRecogCNN()
model.load_state_dict(torch.load("../saved_models/digit_recog_cnn.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

async def predict_digit(image_file):
    img_bytes = await image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        print(logits)
        pred = logits.argmax(dim=1).item()

    return pred