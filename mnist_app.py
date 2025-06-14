import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_model.h5')
    return model

model = load_model()

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 pixel image of a handwritten digit (0–9).")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    st.image(image, caption='Uploaded Image', use_column_width=True)

    prediction = model.predict(img_array)
    st.write(f"### Prediction: **{np.argmax(prediction)}**")
