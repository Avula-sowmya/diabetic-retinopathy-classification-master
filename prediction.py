import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info logs

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras.models import load_model, Model

# =====================================================
# Paths and settings
# =====================================================
CSV_PATH = "labels/train.csv"  # CSV containing image IDs
POSSIBLE_FOLDERS = ["aptos2019-blindness-detection/train_images", "test_images"]  # check which exists
MODEL_PATH = "models/model-unprocessed-unscaled.h5"
SVM_PATH = "models/hybrid-nopca.pkl"
BATCH_SIZE = 32
IMG_SIZE = 224  # Resize images to 224x224

# =====================================================
# Detect data folder automatically
# =====================================================
PATH_TO_DATA = None
for folder in POSSIBLE_FOLDERS:
    if os.path.exists(folder) and os.path.isdir(folder):
        PATH_TO_DATA = folder
        break

if PATH_TO_DATA is None:
    raise FileNotFoundError(f"None of the expected image folders exist: {POSSIBLE_FOLDERS}")

print(f"Using image folder: {PATH_TO_DATA}")

# =====================================================
# Load CSV
# =====================================================
train_df = pd.read_csv(CSV_PATH)
test_df = train_df[['id_code']]  # only image IDs
print(f"Loaded {CSV_PATH} with {len(test_df)} entries")

# =====================================================
# Image preprocessing functions
# =====================================================
def resize_image(image):
    return cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def normalize_image(image):
    image = resize_image(image)
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# =====================================================
# Load CNN model (MobileNetV2 feature extractor)
# =====================================================
model = load_model(MODEL_PATH)
layer_name = 'dense_1'  # layer to extract features from
feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
feature_model.summary()

# =====================================================
# Load pretrained SVM classifier
# =====================================================
with open(SVM_PATH, 'rb') as fid:
    svm = pickle.load(fid)

# =====================================================
# Batch-wise prediction with progress bar
# =====================================================
preds = []

for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Processing batches"):
    batch_ids = test_df['id_code'][i:i+BATCH_SIZE]
    x_batch = []

    for image_id in batch_ids:
        img_path_png = os.path.join(PATH_TO_DATA, f"{image_id}.png")
        img_path_jpg = os.path.join(PATH_TO_DATA, f"{image_id}.jpg")
        if os.path.exists(img_path_png):
            img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            print(f"⚠️  Warning: Image not found, skipping: {image_id}")
            continue  # skip missing images

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x_batch.append(normalize_image(image))

    if len(x_batch) == 0:
        continue  # skip if no images in batch

    x_batch = np.array(x_batch, dtype=np.float32)
    features = feature_model.predict(x_batch, verbose=0)
    preds_batch = svm.predict(features)
    preds.extend(preds_batch)

# =====================================================
# Save predictions
# =====================================================
submission = pd.DataFrame({
    "id_code": test_df["id_code"][:len(preds)],  # match length
    "diagnosis": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")

