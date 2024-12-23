import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from deepface import DeepFace
import numpy as np

st.set_page_config(
    page_title="Face Recognition",
    page_icon="t",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Face Recognition and Emotion Detection")

# Sidebar for uploading a picture
st.sidebar.header("Upload a Picture")
uploaded_file = st.sidebar.file_uploader("Choose a picture to upload", type=["jpg", "png", "jpeg"])
def predict_age(face_image):
    """
    Predicts the age of a person in the given face image using DeepFace.

    Args:
        face_image: The image array containing the face.

    Returns:
        A dictionary containing the predicted age and other relevant information.
    """
    try:
        analysis = DeepFace.analyze(img_path=face_image, actions=['age'], enforce_detection=False)
        predicted_age = analysis[0]['age']
        return {'age': predicted_age}
    except Exception as e:
        st.error(f"Error predicting age: {e}")
        return {'age': 'Unknown'}
if uploaded_file is not None:
    # Load the image file
    image_bytes = uploaded_file.read()
    st.write(f"Uploaded file size: {len(image_bytes)} bytes")

    # Convert to a NumPy array
    image_array = face_recognition.load_image_file(uploaded_file)

    # Convert NumPy array to a PIL Image
    pil_image = Image.fromarray(image_array)
    st.image(pil_image, use_container_width=True, caption="Uploaded Image")

    # Find all faces in the image
    face_locations = face_recognition.face_locations(image_array)
    st.write(f"I found {len(face_locations)} face(s) in this photograph.")

    # Create a draw object
    image_pil = pil_image.copy()
    draw = ImageDraw.Draw(image_pil)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image_array[top:bottom, left:right]

        try:
            
            analysis = DeepFace.analyze(img_path=face_image, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
            st.write(f"Detected Emotion: {dominant_emotion}")

            # Draw bounding box and label
            draw.rectangle([(left, top), (right, bottom)], outline="blue", width=2)
            font = ImageFont.load_default()  # Use default font
            draw.text((left, top - 10), dominant_emotion, fill="blue", font=font)
            
            age_prediction = predict_age(face_image)
            
            objs = DeepFace.analyze(
            img_path= face_image,
            actions=['gender', 'race', 'emotion'],
            enforce_detection=False
        )
            analysis = DeepFace.analyze(img_path=face_image, actions=['race'], enforce_detection=False)
            dominant_race= analysis[0].get('dominant_race', 'Unknown')
          
            st.write(f"Detected race : {dominant_race }")
            

        except Exception as e:
            st.error(f"Error analyzing face: {e}")

    # Display the annotated image with faces and emotions
    st.image(image_pil, caption='Detected Faces and Emotions', use_container_width=True)
