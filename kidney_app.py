# 🚀 Imports
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🧠 Load trained model (no changes needed here)
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Define class labels directly
class_names = ["Normal", "Kidney Stone", "Cyst"]  # Adjust as per your model

# 🌟 Streamlit interface: user uploads image
st.title("Kidney Stone Detector")
uploaded_file = st.file_uploader("Upload a kidney ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 🖼️ Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧼 Preprocess: resize + convert to array
    image = image.resize((64, 64))  # adjust size based on your training
    image_array = np.array(image)
    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Define PCA inside the app (skip loading .pkl)
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(image_flattened)

    # 🔮 Make prediction
    pred = model.predict(X_pca)

    # 📣 Show result
    st.write("Prediction:", class_names[pred[0]])
