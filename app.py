from flask import Flask, request, render_template_string
from PIL import Image, UnidentifiedImageError
import io
import torch
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

# Warm-up
with torch.no_grad():
    model(torch.zeros(1, 3, 224, 224))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

HTML = """
<!DOCTYPE html>
<html>
<body style="font-family: Arial; padding: 40px;">
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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML, result="No file uploaded")

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return render_template_string(
            HTML,
            result="Unsupported file type. Please upload JPG or PNG images."
        )

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except UnidentifiedImageError:
        return render_template_string(
            HTML,
            result="Invalid image file. Please upload a valid JPG or PNG."
        )

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
