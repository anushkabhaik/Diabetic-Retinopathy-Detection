import streamlit as st
import cv2
import numpy as np
import pywt
import pickle
from PIL import Image

IMAGE_SIZE = (256, 256)

# Load models
@st.cache_resource
def load_models():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    return scaler, pca, classifier

# Image preprocessing
def preprocess_image(uploaded_image, size=IMAGE_SIZE):
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, size)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

# Feature extraction
def extract_wavelet_features(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
    return features

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")
    st.write("Upload a fundus image to predict Diabetic Retinopathy (DR).")

    uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image...")

        try:
            scaler, pca, classifier = load_models()
            img = preprocess_image(uploaded_file)
            features = extract_wavelet_features(img)
            features_scaled = scaler.transform([features])
            features_pca = pca.transform(features_scaled)

            prediction = classifier.predict(features_pca)[0]

            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(features_pca)[0]
                if len(proba) == 2:
                    dr_proba = proba[1]
                    no_dr_proba = proba[0]
                else:
                    dr_proba = 1.0 if prediction == 1 else 0.0
                    no_dr_proba = 1.0 if prediction == 0 else 0.0
            else:
                dr_proba = no_dr_proba = None

            if prediction == 1:
                st.error(f"DR Detected. Probability: {dr_proba * 100:.2f}%" if dr_proba is not None else "🩺 DR Detected.")
            else:
                st.success(f"No DR Detected. Probability: {no_dr_proba * 100:.2f}%" if no_dr_proba is not None else "✅ No DR Detected.")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
