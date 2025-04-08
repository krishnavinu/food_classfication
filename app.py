import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load the trained model
MODEL_PATH = 'best_food_classifier.h5'
model = load_model(MODEL_PATH)

# Streamlit UI
st.title("🍔 Food vs Non-Food Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make a prediction
    prediction = model.predict(img_array)[0][0]
    confidence = round(prediction, 4)
    
    # Display results
    st.subheader("Prediction:")
    if confidence > 0.5:
        st.success(f"🍕 The image is classified as **FOOD** with {confidence*100:.2f}% confidence.")
    else:
        st.error(f"🚫 The image is classified as **NON-FOOD** with {(1-confidence)*100:.2f}% confidence.")
