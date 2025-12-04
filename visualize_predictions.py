import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# =============================
# Configuration
# =============================
IMAGE_FOLDER = "train_images"  # Change this to your actual image folder
SUBMISSION_FILE = "submission.csv"

# Load submission.csv
submission = pd.read_csv(SUBMISSION_FILE)

# Mapping DR grade to one example image
examples = {}

# Collect one example per DR grade (0-4)
for row in submission.itertuples():
    grade = row.diagnosis
    if grade in examples:
        continue  # already have one example for this grade
    
    # Try both .png and .jpg
    img_path = os.path.join(IMAGE_FOLDER, f"{row.id_code}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_FOLDER, f"{row.id_code}.jpg")
    if not os.path.exists(img_path):
        continue  # skip if image not found
    
    examples[grade] = img_path
    
    if len(examples) == 5:  # got all grades
        break

# =============================
# Display images
# =============================
plt.figure(figsize=(15, 4))
for i, grade in enumerate(sorted(examples.keys()), 1):
    img_path = examples[grade]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 5, i)
    plt.imshow(image)
    plt.title(f"Predicted: {grade}")
    plt.axis('off')

plt.tight_layout()
plt.show()

