import streamlit as st
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time

# Load the pre-trained model
model = load_model("Models/FaceDetect/best_model.h5")

# Load the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion from an image
def predict_emotion(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

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

        # Draw a rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return frame, predicted_emotion if faces_detected != () else "No Face Detected"

# Streamlit App
st.title("Facial Emotion Analysis")

# Display instructions
st.markdown("Click the button below to start the webcam and analyze facial emotions.")

# Placeholder for video frames
frame_placeholder = st.empty()
emotion_placeholder = st.empty()

# Start the webcam when the button is clicked
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error accessing webcam. Please try again.")
            break

        # Predict emotion on the frame
        frame, emotion = predict_emotion(frame)

        # Display the emotion and the frame
        emotion_placeholder.text(f"Detected Emotion: {emotion}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        frame_placeholder.image(frame, channels="RGB")

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Instruction to stop the webcam
st.markdown("Press 'q' in the webcam window to stop the analysis.")
