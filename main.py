import streamlit as st
from PIL import Image, ImageOps
from classifier import cancerPredict

st.title("Breast Cancer Detector Using Machine Learning")
st.header("Breast Cancer Ultrasound Classification Example")
st.text("Upload a scan for Classification")


uploaded_file = st.file_uploader("Choose a scan result ...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = cancerPredict(image, 'model/keras_model.h5')
    if label == 0:
        st.write("The scan is normal")
    elif label == 1:
        st.write("The scan is malignant")
    else:
        st.write("The scan is benign")
