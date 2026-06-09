import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # -> (N, 32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # -> (N, 64, 24, 24)
        self.pool  = nn.MaxPool2d(2)                   # -> (N, 64, 12, 12)
        self.drop1 = nn.Dropout(0.25)
        # flatten                                       # -> (N, 9216)  [64 * 12 * 12]
        self.fc1   = nn.Linear(9216, 128)              # -> (N, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 10)                # -> (N, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # (N, 32, 26, 26)
        x = torch.relu(self.conv2(x))   # (N, 64, 24, 24)
        x = self.pool(x)                # (N, 64, 12, 12)
        x = self.drop1(x)
        x = torch.flatten(x, 1)        # (N, 9216)
        x = torch.relu(self.fc1(x))     # (N, 128)
        x = self.drop2(x)
        x = self.fc2(x)                 # (N, 10)
        return x


@st.cache_resource
def load_model():
    model = MnistCNN()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
    model.eval()
    return model


def preprocess(image_data):
    """
    Pipeline MNIST officiel :
    1. RGBA -> niveaux de gris
    2. Recadrage sur la bounding box du trace
    3. Resize dans 20x20 (ratio preserve)
    4. Centrage dans 28x28 par centre de masse des pixels
    5. Normalisation MNIST (mean=0.1307, std=0.3081)
    Retourne (tensor (1,1,28,28), img_01 float32 [0,1]) ou None si vide.
    """
    # 1. RGBA -> grayscale
    rgba = Image.fromarray(image_data.astype(np.uint8), mode="RGBA")
    gray = np.array(rgba.convert("L"), dtype=np.float32)  # [0, 255]

    # 2. Bounding box (seuil > 10 pour ignorer le bruit)
    rows = np.any(gray > 10, axis=1)
    cols = np.any(gray > 10, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = gray[rmin:rmax+1, cmin:cmax+1]

    # 3. Resize dans 20x20 en preservant le ratio
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    digit = np.array(
        Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    )

    # 4. Centrage dans 28x28 par centre de masse
    total = digit.sum()
    if total == 0:
        return None
    ys, xs = np.mgrid[0:new_h, 0:new_w]
    cy = (ys * digit).sum() / total
    cx = (xs * digit).sum() / total

    top  = int(round(14 - cy))
    left = int(round(14 - cx))

    canvas = np.zeros((28, 28), dtype=np.float32)
    src_r = max(0, -top)
    dst_r = max(0, top)
    src_c = max(0, -left)
    dst_c = max(0, left)
    h_p = min(new_h - src_r, 28 - dst_r)
    w_p = min(new_w - src_c, 28 - dst_c)
    if h_p > 0 and w_p > 0:
        canvas[dst_r:dst_r+h_p, dst_c:dst_c+w_p] = \
            digit[src_r:src_r+h_p, src_c:src_c+w_p]

    # 5. Normalisation
    img_01 = canvas / 255.0
    normalized = (img_01 - 0.1307) / 0.3081

    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, img_01


# App

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="X")
st.title("MNIST Digit Recognizer")
st.markdown("Dessinez un chiffre (0-9) puis cliquez sur **Predict**.")

model = load_model()

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col_draw, col_result = st.columns([1, 1])

with col_draw:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    col_p, col_c = st.columns(2)
    predict_btn = col_p.button("Predict", type="primary", use_container_width=True)
    if col_c.button("Clear", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

with col_result:
    if predict_btn:
        if canvas_result.image_data is None:
            st.info("Dessinez un chiffre d'abord.")
        else:
            result = preprocess(canvas_result.image_data)
            if result is None:
                st.info("Canvas vide — dessinez un chiffre.")
            else:
                tensor, img_01 = result

                st.markdown("**Ce que voit le modele (28x28) :**")
                st.image(np.clip(img_01, 0, 1), width=140, clamp=True)

                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].numpy()

                pred = int(np.argmax(probs))
                conf = float(probs[pred]) * 100

                st.markdown(f"## Predit : **{pred}**")
                st.markdown(f"Confiance : **{conf:.1f}%**")

                st.bar_chart(
                    pd.DataFrame(
                        {"Probabilite": probs},
                        index=[str(i) for i in range(10)],
                    ),
                    height=250,
                )
