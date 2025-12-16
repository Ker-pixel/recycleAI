import torch
torch.set_num_threads(1)
from flask import Flask, request, render_template_string
from PIL import Image
import io
import torch.nn.functional as F
from torchvision import transforms
from src.model import build_model

MODEL_PATH = "models/resnet18_binary.pth"

app = Flask(__name__)

# Load model once
device = torch.device("cpu")
model = build_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Warm-up inference to avoid first-request timeout
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224)
    model(dummy)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

HTML = """
<!DOCTYPE html>
<html>
<body style="font-family: Arial; padding: 40px;">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Ubuntu:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400;1,500;1,700&display=swap" rel="stylesheet">
<style>
    * {
        font-family: Ubuntu;
        text-align: center;
        font-size: larger;
        background-color: rgb(18, 95, 18);
        color: wheat;
    }
</style>
<h1>Recycling Classifier</h1>

<form action="/predict" method="POST" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button type="submit">Analyze</button>
</form>

{% if result %}
<h2>Result: {{ result }}</h2>
{% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    if conf.item() < 0.55:
        result = "not recyclable (no)"
    else:
        result = "recyclable (yes)" if pred.item() == 1 else "not recyclable (no)"

    return render_template_string(HTML, result=result)