import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r"D:\DS_PRACTICE\26-06-2026\HappySadCNN\emotion_model.h5"
    )
    return model

model = load_model()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Happy or Sad Face Detection",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Happy / 😢 Sad Face Detection")

st.write("Upload an image and the AI model will predict whether the face is Happy or Sad.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize according to model input
    img = image.resize((200, 200))

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = prediction[0][0]

    if probability > 0.5:
        emotion = "Sad 😢"
        confidence = probability
    else:
        emotion = "Happy 😊"
        confidence = 1 - probability

    st.markdown("---")

    st.subheader("Prediction")

    st.success(f"**Emotion:** {emotion}")

    st.info(f"Confidence: **{confidence*100:.2f}%**")