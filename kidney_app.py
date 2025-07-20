import streamlit as st
import numpy as np
import pickle
from PIL import Image

# 🔍 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🩺 App layout
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🖼 Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 📐 Resize and preprocess
    image = image.resize((64, 64))  # Same size used during training
    image_array = np.array(image)

    # Convert to grayscale if RGB
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 🔮 Predict class index
    pred = model.predict(image_flattened)

    # 🏷️ Decode class index (hardcoded labels — safe and readable)
    class_names = ["Normal", "Kidney Stone", "Cyst"]  # Adjust if needed
    label = class_names[pred[0]]

    # ✅ Show result
    st.success(f"Prediction: {label}")
