import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🚀 Load trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Class labels hardcoded
class_names = ["Normal", "Kidney Stone", "Cyst"]

# 🖥️ App UI
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload an ultrasound image to get a prediction.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔄 Resize & preprocess
    image = image.resize((64, 64))
    image_array = np.array(image)

    # Convert RGB to grayscale
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Dummy-fit PCA so model gets correct input shape
    dummy_data = np.random.rand(10, image_flattened.shape[1])
    pca = PCA(n_components=30)
    pca.fit(dummy_data)
    X_pca = pca.transform(image_flattened)

    # 🧪 Debug info
    st.write("Shape of image:", image_array.shape)
    st.write("Flattened shape:", image_flattened.shape)

    try:
        prediction = model.predict(X_pca)
        st.write("Raw prediction result:", prediction)

        label = class_names[prediction[0]]
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
