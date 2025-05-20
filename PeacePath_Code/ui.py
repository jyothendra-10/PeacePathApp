import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
from deepface import DeepFace
import cv2
import numpy as np
from datetime import datetime
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import google.generativeai as genai
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import warnings
import speech_recognition as sr
warnings.filterwarnings('ignore')

# Load text emotion model
model = load_model('emotion_detection_model.h5')
lang = "English"

# Initialize emotion history
if "latest_emotion" not in st.session_state:
    st.session_state.latest_emotion = None

# DeepFace analysis function
def analyze_emotion(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    res = DeepFace.analyze(img_cv, actions=['emotion'])
    data_dict = res[0]['emotion']
    max_pair = max(data_dict.items(), key=lambda x: x[1])
    return max_pair[0]

# Load text data and model
data = pd.read_csv("Text Emotion Detection/EX 2/train.txt", sep=';')
data.columns = ["Text", "Emotions"]
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

json_file = open("Text Emotion Detection/EX 2/model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Text Emotion Detection/EX 2/model_weights.h5")

st.set_page_config(page_title='Emotion Detection', layout='wide', page_icon=":smiley:", initial_sidebar_state="expanded")

with st.sidebar:
    selected = option_menu("Peacepath", ["Home", 'Text', 'Video', 'Speech', 'Chatbot'], 
        icons=['house','card-text','card-image','headset','chat'], menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "red"}})

if selected == 'Home':
    st.markdown("<h1 style='text-align: center;font-size:60px;color:red;'>Emotion Detection</h1>", unsafe_allow_html=True)
    st.image('https://cdn.dribbble.com/userupload/20930023/file/original-3d90fb09f5268a362e6ee1c01e0bbb48.gif', width=800)

if selected == 'Text':
    st.markdown("<h1 style='text-align: center;font-size:60px;color:red;'>Text Emotion Detection</h1>", unsafe_allow_html=True)
    text = st.text_input('Enter Text')
    col1, col2, col3 = st.columns([3, 3, 1])
    if col2.button('Predict', type='primary'):
        if text:
            input_sequence = tokenizer.texts_to_sequences([text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = loaded_model.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            st.session_state.latest_emotion = f"Text Emotion: {predicted_label}"
            st.success(st.session_state.latest_emotion)
        else:
            st.error('Please enter text')

if selected == 'Video':
    st.markdown("<h1 style='text-align: center;font-size:60px;color:red;'>Facial Emotion Detection</h1>", unsafe_allow_html=True)
    stframe = st.empty()
    emotion_placeholder = st.empty()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            for result in results:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                emotion = result['dominant_emotion']
                st.session_state.latest_emotion = f"Video Emotion: {emotion}"
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(rgb_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        except Exception as e:
            emotion_placeholder.error(f"Error detecting emotion: {str(e)}")
    camera.release()
    cv2.destroyAllWindows()

if selected == 'Speech':
    st.markdown("<h1 style='text-align: center;font-size:60px;color:red;'>Speech Emotion Detection</h1>", unsafe_allow_html=True)

    def transcribe_audio():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio_data = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results; {e}"

    st.markdown('----')
    col1, col2, col3 = st.columns([3, 3, 1])
    if col2.button("Start Recording", type='primary'):
        text = transcribe_audio()
        if text == "Could not understand audio":
            st.write("Please speak clearly and try again")
        else:
            st.write(f"Text: {text}")
            input_sequence = tokenizer.texts_to_sequences([text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = loaded_model.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            st.session_state.latest_emotion = f"Speech Emotion: {predicted_label}"
            st.success(st.session_state.latest_emotion)

if selected == 'Chatbot':
    api_key = "AIzaSyCEHqEUnURAuH54Tng8IjlWSR6LyzzEpCI"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    st.title("PeacePath your Mental health companion 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.latest_emotion:
        initial_prompt = f"The user recently expressed this emotion: {st.session_state.latest_emotion}. Please respond empathetically and offer support."
        response = model.generate_content([initial_prompt])
        bot_response = response.text
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.markdown(f"**PeacePath:** {bot_response}")
        st.session_state.latest_emotion = None  # Clear after using

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("I'm here for you — a friend who listens, understands, and gently guides. Don't hesitate to open up.")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = model.generate_content([user_input])
        bot_response = response.text
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        with st.chat_message("bot"):
            st.markdown(bot_response)
