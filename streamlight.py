import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained CNN model
model = load_model("C:/Users/manoj/Downloads/cnn_model.h5")

# Define class labels exactly as per your dataset
class_names = ["closed", "no_yawn", "open", "yawn"]

st.title("🧠 CNN Image Classifier")
st.write("Upload an image and the model will predict one of: closed, no_yawn, open, yawn")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize to match CNN input)
    img = image.load_img(uploaded_file, target_size=(224, 224))  # adjust if your model uses different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Show result
    st.write(f"### Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
