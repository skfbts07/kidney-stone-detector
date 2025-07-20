import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🎯 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Define class labels directly (replace with your actual labels)
class_names = ["Normal", "Kidney Stone", "Cyst"]

# 🎨 App interface
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🖼 Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧼 Preprocess image
    image = image.resize((64, 64))  # Use same size as during model training
    image_array = np.array(image)
    
    # If image is RGB, convert to grayscale (optional — only if model expects grayscale)
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Apply PCA inside the app
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(image_flattened)

    # 🔮 Make prediction
    pred = model.predict(X_pca)
    
    # 📣 Show readable result
    st.success(f"Prediction: {class_names[pred[0]]}")
