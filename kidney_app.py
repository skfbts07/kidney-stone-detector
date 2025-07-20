import streamlit as st
import numpy as np
import pickle
from PIL import Image
import sklearn, sys

# 🔍 Version info for debugging
st.write("Python version:", sys.version)
st.write("scikit-learn version:", sklearn.__version__)

# 🚀 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 📊 Load PCA transformer
with open("pca_transform.pkl", "rb") as f:
    pca = pickle.load(f)

# 🏷️ Load class labels
with open("class_labels.pkl", "rb") as f:
    class_names = pickle.load(f)

# 🎨 App layout
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚙️ Resize and preprocess
    image = image.resize((64, 64))
    image_array = np.array(image)

    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)  # Convert RGB to grayscale

    image_flattened = image_array.flatten().reshape(1, -1)

    # 🔁 Apply PCA
    try:
        X_pca = pca.transform(image_flattened)
        prediction = model.predict(X_pca)
        label = class_names[prediction[0]]
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Flattened shape:", image_flattened.shape)
