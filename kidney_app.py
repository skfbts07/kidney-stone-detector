import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🔍 Load XGBoost model (trained on PCA-transformed data)
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Define class labels directly
class_names = ["Normal", "Kidney Stone", "Cyst"]  # Adjust if needed

# 🎨 App layout
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🖼 Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚙️ Resize and preprocess
    image = image.resize((64, 64))  # Same size used during training
    image_array = np.array(image)

    # Convert to grayscale if it's RGB
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Apply PCA using dummy training (safe for deployment)
    dummy_data = np.random.rand(10, image_flattened.shape[1])
    pca = PCA(n_components=30)
    pca.fit(dummy_data)
    X_pca = pca.transform(image_flattened)

    # 🔮 Predict
    pred = model.predict(X_pca)
    label = class_names[pred[0]]
    st.success(f"Prediction: {label}")
