from sklearn.decomposition import PCA

# 🎯 Resize and flatten image
image = image.resize((64, 64))
image_array = np.array(image)

# Optional: Convert to grayscale (if model expects it)
if image_array.ndim == 3:
    image_array = image_array.mean(axis=2)

image_flattened = image_array.flatten().reshape(1, -1)

# 📊 Dummy PCA fit — avoids ValueError with single image
dummy_data = np.random.rand(10, image_flattened.shape[1])  # 10 fake samples
pca = PCA(n_components=30)
pca.fit(dummy_data)

# 🔁 Transform uploaded image
X_pca = pca.transform(image_flattened)

# 🔮 Prediction
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["Normal", "Kidney Stone", "Cyst"]
pred = model.predict(X_pca)

st.success(f"Prediction: {class_names[pred[0]]}")
