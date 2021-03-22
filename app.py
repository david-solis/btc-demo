import keras
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
MODEL_FILE = 'model.h5'

st.title("Brain Tumor Classifier")
st.header("End-to-end Learning")

uploaded_file = st.file_uploader("Please upload a brain MRI scan", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI brain scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Load model
    model = keras.models.load_model(MODEL_FILE)
    # Convert to the image format expected by the model
    image = ImageOps.fit(image, IMG_SIZE, Image.ANTIALIAS).convert('L')
    image_array = np.expand_dims(np.array(image), axis=2)
    image_array = image_array.reshape((1,) + image_array.shape)

    prediction = model.predict(image_array)
    label = np.argmax(prediction)

    st.write(("Tumor" if label else "No tumor") + " detected")
