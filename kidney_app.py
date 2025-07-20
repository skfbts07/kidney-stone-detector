import streamlit as st
import numpy as np
import pickle
from PIL import Image
import sklearn, sys

# 🧠 Version info (for debugging during deployment)
st.write("Python version:", sys.version)
st.write("scikit-learn version:", sklearn.__version__)

# 🔍 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 📊 Load PCA transformer
with open("pca_transform.pkl", "rb") as f:
    pca = pickle.load(f)

# 🏷️ Load class labels
with open("class_labels.pkl", "rb") as f:
    class_names = pickle.load(f)

# 🎨 App interface
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a diagnosis.")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🖼 Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧼 Resize and flatten
    image = image.resize((64, 64))
    image_array = np.array(image)

    # Convert to grayscale if it's RGB
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 🔁 Apply trained PCA
    X_pca = pca.transform(image_flattened)

    # 🔮 Make prediction
    pred = model.predict(X_pca)
    st.success(f"Prediction: {class_names[pred[0]]}")
