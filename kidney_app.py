import streamlit as st
import numpy as np
import pickle
from PIL import Image

# 🔍 Load trained model only
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🩺 Streamlit UI
st.title("Kidney Stone Detector")
st.markdown("Upload an ultrasound image to get a prediction.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((64, 64))
    image_array = np.array(image)

    # Convert RGB to grayscale if needed
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 🔮 Predict directly
    pred = model.predict(image_flattened)

    st.success(f"Prediction: Class {pred[0]}")
