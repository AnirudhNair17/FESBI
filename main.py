import streamlit as st
import tensorflow as tf  # Use TensorFlow for model loading
import requests
from streamlit_lottie import st_lottie
from PIL import Image, ImageOps
import numpy as np
import os
import uuid
import webbrowser  # Import the webbrowser module

# URLs for social media and About section
LINKEDIN_URL = "https://www.linkedin.com/company/fesforcommons/"
GMAIL_URL = "mailto:ashok@fes.org.in"
YOUTUBE_URL = "https://www.youtube.com/@ecologicalsecurity"
ABOUT_URL = "https://fes.org.in/about/our-mission"

# Load Lottie animation
lottie_url = "https://lottie.host/50849054-36b4-4b5d-8cd4-772f0ec00d5d/K9KMg8R09O.json"
lottie_json = requests.get(lottie_url).json() if requests.get(lottie_url).status_code == 200 else {}

st_lottie(lottie_json, width=300, height=300)
np.set_printoptions(suppress=True)

# Load the model with error handling
try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if the model fails to load

# Load the labels
try:
    with open("labels.txt", "r") as file:
        class_names = [line.strip()[2:] for line in file.readlines()]
except Exception as e:
    st.error(f"Error loading labels file: {e}")
    st.stop()

def classify_bird(image_path):
    """Classifies a bird image and displays the results in Streamlit."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        prediction = model.predict(data)
        top_class_idx = np.argmax(prediction[0])
        top_confidence = prediction[0][top_class_idx]
        top_class_name = class_names[top_class_idx]

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Predicted Bird:** {top_class_name} (Confidence: {top_confidence:.4f})")

        if top_confidence < 0.97:
            st.write("Top 3 Possible Classes:")
            top_classes = np.argsort(prediction[0])[::-1][:3]
            for i, class_idx in enumerate(top_classes):
                st.write(f"{i + 1}. {class_names[class_idx]} - Confidence: {prediction[0][class_idx]:.4f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Ensure directory for reference images exists
reference_images_folder = "reference_images"
os.makedirs(reference_images_folder, exist_ok=True)

if __name__ == "__main__":
    st.title("Bird Identification App")
    image_file = st.file_uploader("Upload an image of a bird:")
    
    if image_file:
        classify_bird(image_file)

        # Save feedback and reference images
        feedback = st.radio("Was the classification accurate?", ["Like", "Dislike"])
        
        if feedback == "Like" and st.button("Submit Feedback"):
            st.success("Thank you for the feedback! 🐦")
        
        elif feedback == "Dislike":
            correct_name = st.text_input("Enter the correct species name:")
            if correct_name and st.button("Submit Correction"):
                image_path = os.path.join(reference_images_folder, f"{correct_name}.jpg")
                with open(image_path, "wb") as f:
                    f.write(image_file.getvalue())
                st.success(f"Reference image for {correct_name} saved!")

            if st.button("I don't know"):
                unknown_filename = f"unknown_{uuid.uuid4().hex[:8]}.jpg"
                unknown_path = os.path.join(reference_images_folder, unknown_filename)
                with open(unknown_path, "wb") as f:
                    f.write(image_file.getvalue())
                st.success("Unknown image saved for further analysis.")

    # Social Media Links
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("LinkedIn"):
            webbrowser.open_new_tab(LINKEDIN_URL)
    with col2:
        if st.button("Gmail"):
            webbrowser.open_new_tab(GMAIL_URL)
    with col3:
        if st.button("YouTube"):
            webbrowser.open_new_tab(YOUTUBE_URL)
    with col4:
        if st.button("About"):
            webbrowser.open_new_tab(ABOUT_URL)
