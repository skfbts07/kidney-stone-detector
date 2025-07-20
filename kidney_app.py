import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🚀 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Class labels defined in code
class_names = ["Normal", "Kidney Stone", "Cyst"]

# 🖥️ App layout
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 📐 Resize & convert to grayscale
    image = image.resize((64, 64))
    image_array = np.array(image)

    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Dynamic PCA setup to avoid crashes
    n_features = image_flattened.shape[1]
    n_components = max(1, min(30, n_features - 1))  # Safe: at least 1

    dummy_data = np.random.rand(10, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(dummy_data)
    X_pca = pca.transform(image_flattened)

    # 🧪 Debug info
    st.write("Flattened shape:", image_flattened.shape)
    st.write("Using PCA components:", n_components)

    try:
        prediction = model.predict(X_pca)
        label = class_names[prediction[0]]
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
