import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA

# 🚀 Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Hardcoded class labels
class_names = ["Normal", "Kidney Stone", "Cyst"]

# 🎨 App layout
st.set_page_config(page_title="Kidney Stone Detector", layout="centered")
st.title("🩺 Kidney Stone Detector")
st.markdown("Upload a kidney ultrasound image to get a prediction.")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 🖼 Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚙️ Resize and preprocess
    image = image.resize((64, 64))
    image_array = np.array(image)

    # Convert to grayscale if RGB
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2)

    image_flattened = image_array.flatten().reshape(1, -1)

    # 📊 Safe PCA with dynamic component count
    n_features = image_flattened.shape[1]
    n_components = min(30, n_features)

    dummy_data = np.random.rand(10, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(dummy_data)
    X_pca = pca.transform(image_flattened)

    # 🧪 Debugging outputs
    st.write("Image shape:", image_array.shape)
    st.write("Flattened shape:", image_flattened.shape)
    st.write("PCA components used:", n_components)

    try:
        prediction = model.predict(X_pca)
        st.write("Raw prediction result:", prediction)
        label = class_names[prediction[0]]
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
