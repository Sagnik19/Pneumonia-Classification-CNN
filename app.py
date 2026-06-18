import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from model import predict_image

st.set_page_config(
    page_title="Pneumonia Detection using CNN",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection using CNN")
st.write("Upload a Chest X-Ray image to predict whether the patient has Pneumonia or is Normal.")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    image = image.convert("L")
    image = image.resize((224, 224))

    img_array = img_to_array(image)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    label, confidence = predict_image(img_array)

    st.subheader("Prediction Result")

    if label == "PNEUMONIA":
        st.error(f"🦠 Prediction: {label}")
    else:
        st.success(f"✅ Prediction: {label}")

    st.metric(
        label="Confidence Score",
        value=f"{confidence:.4f}"
    )

st.markdown("---")
st.caption("Developed by Sagnik Das | CNN-based Pneumonia Detection System")