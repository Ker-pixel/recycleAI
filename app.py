import torch
torch.set_num_threads(1)

from flask import Flask, request, render_template_string
from PIL import Image, UnidentifiedImageError
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

EXPLANATIONS = {
    "recyclable_item": {
        "label": "Geri dönüştürülebilir ürün",
        "text": "Bu ürün geri dönüştürülebilir görünüyor. Gerekirse temizleyin ve yerel geri dönüşüm kurallarına göre atın."
    },
    "non_recyclable_item": {
        "label": "Geri dönüştürülemez ürün",
        "text": "Bu ürün normal geri dönüşüme atılmamalıdır. Yerel atık kurallarına göre imha edin."
    },
    "unknown": {
        "label": "Bilinmeyen ürün",
        "text": "Ürün güvenilir şekilde tanımlanamadı. Emin değilseniz geri dönüşüme atmayın."
    }
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Geri Dönüşüm Sınıflandırıcısı</title>

  <!-- Google Font -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">

  <!-- Favicon (PNG is fine) -->
  <link rel="icon" type="image/png" href="{{ url_for('static', filename='favicon.png') }}">

  <style>
    body {
      font-family: 'Ubuntu', sans-serif;
      background-color: rgb(18, 95, 18);
      color: wheat;
      text-align: center;
      padding: 40px;
    }

    h1 {
      font-size: 90px;
      margin: 0;
      line-height: 1.1;
    }

    h2 {
      font-size: 50px;
      margin-top: 40px;
    }

    form {
      margin-top: 50px;
    }

    input[type="file"] {
      font-size: 20px;
      margin-bottom: 20px;
    }

    button {
      font-size: 24px;
      padding: 10px 24px;
      cursor: pointer;
    }
    .card {
    background-color: rgba(0, 0, 0, 0.25);
    border-radius: 16px;
    padding: 30px;
    margin-top: 40px;
    display: inline-block;
    max-width: 600px;
    }

    .result-yes {
      color: #9cff9c;
    }

    .result-no {
      color: #ffb3b3;
    }
    input[type="file"] {
      display: none;
    }

    .upload-label {
      display: inline-block;
      background-color: #2f7d2f;
      padding: 14px 30px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 22px;
    }
  </style>
</head>

<body>

  <h1>Geri Dönüşüm</h1>
  <h1>Sınıflandırıcısı</h1>

  <form action="/predict" method="POST" enctype="multipart/form-data">
    <label class="upload-label">
      Görsel seç
      <input type="file" name="file" required>
    </label>
    <br><br>
    <button type="submit">Analiz et</button>
  </form>

  {% if result %}
    <div class="card">
      <h2 class="{{ 'result-yes' if 'yes' in result else 'result-no' }}">
        {{ result }}
      </h2>
      <p><strong>Tespit edilen ürün:</strong> {{ item }}</p>
      <p><strong>Güven:</strong> {{ confidence }}</p>
      <p><strong>Nasıl geri dönüştürülür:</strong> {{ explanation }}</p>
    </div>
  {% endif %}

  <p style="opacity: 0.8; font-size: 16px; margin-top: 40px;">
    Sonuçlar görsel görünüme dayalıdır. Yerel geri dönüşüm kuralları değişiklik gösterebilir.
  </p>

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
        return render_template_string(HTML, result="Dosya yüklenmedi")

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return render_template_string(
            HTML,
            result="Desteklenmeyen dosya türü. Lütfen JPG veya PNG yükleyin."
        )

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return render_template_string(
            HTML,
            result="Geçersiz görsel dosyası. Lütfen geçerli bir JPG veya PNG yükleyin."
        )

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item()
    prediction = pred.item()

    if confidence < 0.55:
        item_key = "unknown"
        result = "geri dönüştürülemez (hayır)"
    else:
        if prediction == 1:
            item_key = "recyclable_item"
            result = "geri dönüştürülebilir (evet)"
        else:
            item_key = "non_recyclable_item"
            result = "geri dönüştürülemez (hayır)"

    return render_template_string(
        HTML,
        result=result,
        item=EXPLANATIONS[item_key]["label"],
        explanation=EXPLANATIONS[item_key]["text"],
        confidence=f"{confidence:.2f}"
    )
