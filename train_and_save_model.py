import os
import numpy as np
import cv2
import pywt
import pickle
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FUNDUS_DIR = r"C:\Users\anush\Downloads\Diabetic Retinopathy Detection\diaretdb0_v_1_1\resources\images\diaretdb0_fundus_images"
MASK_DIR = r"C:\Users\anush\Downloads\Diabetic Retinopathy Detection\diaretdb0_v_1_1\resources\images\example_images"
IMAGE_SIZE = (256, 256)

def preprocess_image(image_path, size=IMAGE_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be read.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.resize(img_gray, size)
    return img_gray

def extract_wavelet_features(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
    return features

def has_lesion_mask(image_filename):
    base = os.path.splitext(image_filename)[0]
    possible_masks = [f for f in os.listdir(MASK_DIR) if base in f]
    return len(possible_masks) > 0

def load_data():
    features = []
    labels = []

    for fname in os.listdir(FUNDUS_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(FUNDUS_DIR, fname)
            try:
                img = preprocess_image(image_path)
                feat = extract_wavelet_features(img)
                label = 1 if has_lesion_mask(fname) else 0
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {fname}: {e}")

    return np.array(features), np.array(labels)

def main():
    print("Loading and processing dataset...")
    X, y = load_data()

    label_dist = Counter(y)
    print(f"Label distribution: {label_dist}")

    if len(set(y)) < 2:
        raise ValueError("Only one class found in dataset. At least two classes (0 and 1) are required to train a classifier.")

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Applying PCA...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%")

    print("Saving trained models...")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open("classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Training complete. Models saved successfully.")

if __name__ == "__main__":
    main()
