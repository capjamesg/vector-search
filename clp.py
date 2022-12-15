import os

import clip
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

HOME_DIR = "/Users/james/Desktop"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

images = []

text = clip.tokenize(["london"]).to(device)

# get all .jpeg items
for item in os.listdir(HOME_DIR):
    if item.lower().endswith((".jpg", ".jpeg", ".png")):
        image = (
            preprocess(Image.open(os.path.join(HOME_DIR, item)))
            .unsqueeze(0)
            .to(device)
        )
        images.append((item, image))
    else:
        continue

data = []

# get the cosine similarity
for i in images:
    with torch.no_grad():
        image_features = model.encode_image(i[1])
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(i[1], text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    data.append(
        {
            "image": i[0],
            "cosine": cosine_similarity(image_features, text_features)
        }
    )

data = sorted(data, key=lambda x: x[1][0], reverse=True)

for i in data:
    print(i)