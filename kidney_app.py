import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load trained components
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
import pickle

with open("pca_transform.pkl", "rb") as f:
    pca = pickle.load(f)
class_names = joblib.load("class_labels.pkl")

st.set_page_config(page_title="Kidney Stone Detection", layout="centered")
st.title("🩺 Kidney Stone Classifier")
st.write("Upload a kidney scan image to predict stone presence.")

# Upload image
uploaded_file = st.file_uploader("Choose a kidney scan image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    arr = np.array(image) / 255.0
    flat = arr.reshape(1, -1)
    reduced = pca.transform(flat)
    pred = model.predict(reduced)

    st.success(f"🧠 Prediction: **{class_names[pred[0]]}**")
