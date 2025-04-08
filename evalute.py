import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("food_classifier.h5")

# Define evaluation dataset path
eval_dir = "dataset/evaluation"

# Load and preprocess evaluation images
def load_images(directory):
    images = []
    labels = []
    classes = sorted(os.listdir(directory))  # Get class names (food, non-food)
    
    for label, category in enumerate(classes):
        category_path = os.path.join(directory, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (150, 150))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels), classes

# Load images and labels
eval_images, eval_labels, class_names = load_images(eval_dir)

# Predict
predictions = model.predict(eval_images)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = np.mean(predicted_labels == eval_labels) * 100
print(f"Evaluation Accuracy: {accuracy:.2f}%")

# Show sample predictions
for i in range(5):  # Show first 5 predictions
    label = "FOOD 🍕" if predicted_labels[i] == 1 else "NON-FOOD 🚫"
    print(f"Image {i+1}: {label}")
