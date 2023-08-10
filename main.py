import streamlit as st
import keras
import webbrowser
import requests
from streamlit_lottie import st_lottie
from PIL import Image, ImageOps
import numpy as np
import os
import geocoder
import pandas as pd

url = requests.get("https://lottie.host/50849054-36b4-4b5d-8cd4-772f0ec00d5d/K9KMg8R09O.json")
url_json = dict()

if url.status_code == 200:
    url_json = url.json()
else:
    print("error")

animation_width = 300
animation_height = 300

st_lottie(url_json, width=animation_width, height=animation_height)
np.set_printoptions(suppress=True)

# Load the model
model = keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def classify_bird(image_path):
    """Classifies a bird image and displays the results in a Streamlit UI."""

    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predicts the model
    prediction = model.predict(data)

    # Get the top predicted class and its confidence score
    top_class_idx = np.argmax(prediction[0])
    top_confidence = prediction[0][top_class_idx]
    top_class_name = class_names[top_class_idx].strip()[2:]

    st.title("Bird Classification App")

    st.image(image, caption="Uploaded Image:")
    st.write("Predicted Bird Class:")

    if top_confidence > 0.97:
        st.write(f"{top_class_name} - Confidence Score: {top_confidence:.4f}")
    else:
        st.write("Top 3 Possible Classes:")

        # Get the top three predicted classes and their corresponding confidence scores
        top_classes = np.argsort(prediction[0])[::-1][:3]
        top_confidences = prediction[0][top_classes]

        for i, class_idx in enumerate(top_classes):
            class_name = class_names[class_idx].strip()[2:]
            confidence_score = top_confidences[i]
            st.write(f"{i + 1}. {class_name} - Confidence Score: {confidence_score:.4f}")


reference_images_folder = "reference_images"
os.makedirs(reference_images_folder, exist_ok=True)

if __name__ == "__main__":
    st.title("Bird Classification App")
    st.write("This app classifies bird images using a deep learning model.")

    image_file = st.file_uploader("Upload an image of a bird:")
    if image_file is not None:
        # Clear previous content and reset layout
        st.empty()

        # Display the Lottie animation and the uploaded image
        classify_bird(image_file)

        # Process the image for prediction
        image = Image.open(image_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        prediction = model.predict(data)
        confidence_threshold = 0.97

        if prediction[0][np.argmax(prediction[0])] < confidence_threshold:
            st.write("Top 3 Possible Classes:")
            prediction = model.predict(data)

            # Get the top three predicted classes and their corresponding confidence scores
            top_classes = np.argsort(prediction[0])[::-1][:3]
            top_confidences = prediction[0][top_classes]

            click_counts = [0, 0, 0]  # Initialize click counts for each option

            for i, class_idx in enumerate(top_classes):
                class_name = class_names[class_idx].strip()[2:]
                confidence_score = top_confidences[i]

                # Display a progress bar to represent the intensity of clicks
                progress_bar = st.progress(click_counts[i] / 10.0)  # Limit to 10 clicks

                if st.button(f"{i + 1}. {class_name} - Confidence Score: {confidence_score:.4f}"):
                    st.write(f"Thank you for selecting the correct class: {class_name}.")
                    click_counts[i] += 1
                    progress_bar.progress(click_counts[i] / 10.0)
                    # Automatically save reference image with the selected class name
                    reference_image_path = os.path.join(reference_images_folder, f"{class_name}.jpg")
                    with open(reference_image_path, "wb") as f:
                        f.write(image_file.getvalue())
                    st.write(f"Reference image for {class_name} saved.")
                    break
            else:  # This else block is executed if the for loop completes without a break
                st.write("No class selected. Click 'I don't know' to save the image.")
        else:
            st.write("Feedback:")
            feedback_option = st.radio("Did you find the classification accurate?", ("Like", "Dislike"))

            if feedback_option == "Like":
                if st.button("Submit Feedback"):
                    st.write("Thank you for the feedback! Enjoy birding.")
            elif feedback_option == "Dislike":
                st.write("Sorry for the inconvenience. Please provide the correct species name:")
                correct_name = st.text_input("Correct Species Name:")

                if correct_name:
                    st.write(f"Thank you for the feedback. We will improve our model for {correct_name}.")

                    # Automatically save reference image with corrected name
                    if image_file is not None:
                        reference_image_path = os.path.join(reference_images_folder, f"{correct_name}.jpg")
                        with open(reference_image_path, "wb") as f:
                            f.write(image_file.getvalue())
                        st.write(f"Reference image for {correct_name} saved.")

                # Add an "I don't know" button
                if st.button("I don't know"):
                    st.write("Thank you for your response. The image has been saved for further analysis.")
                    unknown_image_path = os.path.join(reference_images_folder, "unknown.jpg")
                    with open(unknown_image_path, "wb") as f:
                        f.write(image_file.getvalue())
                    st.write("Unknown image saved.")


