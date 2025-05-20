import streamlit as st
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time

# Load the pre-trained model
model = load_model("Python/best_model.h5")

# Load the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Python/haarcascade_frontalface_default.xml')

# Function to capture photo and predict emotion
def capture_and_predict_emotion():
    cap = cv2.VideoCapture(0)
    time.sleep(5)  # Wait for 5 seconds to capture the photo
    ret, test_img = cap.read()
    cap.release()

    if not ret:
        st.error("Error capturing photo. Please try again.")
        return

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        predicted_emotion = emotions[max_index]

        st.success(f"Emotion Detected: {predicted_emotion}")

    #st.image(test_img, caption="Captured Photo", use_column_width=True)

# Streamlit App
st.title("Facial Emotion Analysis")

# Display instructions and capture button
st.markdown("Click the button below to capture a photo and analyze the facial emotion.")

# Add a button to capture again
if st.button("Capture Again"):
    capture_and_predict_emotion()
