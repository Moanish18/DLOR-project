import os

# URL of the file to download
url = "https://dlor-project.s3.us-east-1.amazonaws.com/VGG19.h5"

# File name to save as
file_name = "VGG19.h5"

# Download the file using wget
os.system(f"wget -O {file_name} {url}")

# Save the file name to a variable
model_file = file_name

print(f"Model file saved as: {model_file}")

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained VGG19 model
@st.cache_resource
def load_vgg19_model():
    return tf.keras.models.load_model(model_file)  # Ensure your model is named 'VGG19.h5'

model = load_vgg19_model()

# Function to preprocess the image
def preprocess_image(img, target_size=(32, 32)):
    img = img.resize(target_size)  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Streamlit UI
st.title("🖼️ VGG19 Image Classifier")
st.write("Upload an image and let the model predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed_img)

    # Get the class label
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"🧐 Predicted Class: **{predicted_class}**")

    # Optional: Display raw prediction probabilities
    st.write("Raw Prediction Probabilities:")
    st.write(prediction)
