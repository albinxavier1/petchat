import os
import streamlit as st
import numpy as np
from PIL import Image
import io
from moviepy.editor import ImageSequenceClip, AudioFileClip
from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile
import glob
import requests
import cv2

# Load environment variables
load_dotenv()

# Clean up old temporary files
def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for pattern in ["streamlit_audio_*.mp3", "streamlit_video_*.mp4"]:
        for filename in glob.glob(os.path.join(temp_dir, pattern)):
            try:
                os.remove(filename)
            except Exception as e:
                st.warning(f"Failed to remove {filename}: {e}")

# Run cleanup at the start
cleanup_temp_files()

# Initialize the LLM and Assistant
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("API_KEY"))
assistant = Assistant(llm=llm, tools=[DuckDuckGo()], show_tool_calls=False)

# Streamlit app title
st.title("Talking Animal Avatar App")

# Define the API URL and headers for Hugging Face
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

# Function to query the image generation API
@st.cache_data
def query_image_api(prompt):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Error querying image API: {e}")
        return None

# Image generation based on user prompt
image_prompt = st.text_input("Enter a prompt for image generation:")
generated_image = None

if image_prompt:
    with st.spinner("Generating image..."):
        image_bytes = query_image_api(image_prompt)
        if image_bytes:
            generated_image = Image.open(io.BytesIO(image_bytes))
            st.image(generated_image, caption="Generated Image", use_column_width=True)

# Function to listen to user input
@st.cache_resource
def get_recognizer():
    return sr.Recognizer()

def listen_to_user():
    recognizer = get_recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=5)
        
        text = recognizer.recognize_google(audio)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand that.")
    except sr.RequestError:
        st.error("Sorry, there was an error with the speech recognition service.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    return None

# Function to generate response using the Assistant
def generate_response(prompt):
    try:
        response = assistant.run(prompt)
        # Ensure the response is a string
        return ' '.join(response) if hasattr(response, '__iter__') else str(response)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Function to convert text to speech
def text_to_speech(text):
    try:
        # Ensure text is a string, not a generator
        if not isinstance(text, str):
            text = ' '.join(text) if hasattr(text, '__iter__') else str(text)
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="streamlit_audio_")
        tts = gTTS(text=text, lang='en')
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

# Function to create a simple mouth animation using OpenCV
def create_mouth_animation(image, audio_duration):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(cv_image, 1.1, 4)
    
    frames = []
    fps = 30
    total_frames = int(audio_duration * fps)
    
    for i in range(total_frames):
        frame = cv_image.copy()
        
        for (x, y, w, h) in faces:
            # Define mouth region (adjust these values as needed)
            mouth_y = y + int(h * 0.7)
            mouth_h = int(h * 0.2)
            
            # Simple oscillating mouth movement
            mouth_open = int(10 * np.sin(i * 0.5) + 10)
            
            # Draw mouth
            cv2.ellipse(frame, (x + w//2, mouth_y + mouth_h//2), 
                        (w//4, mouth_open), 0, 0, 180, (0, 0, 0), -1)
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    return frames

# Function to create talking avatar video with basic lip-sync
def create_talking_avatar(image, audio_path):
    try:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="streamlit_video_")
        
        # Load audio clip
        audio_clip = AudioFileClip(audio_path)
        
        # Create mouth animation
        frames = create_mouth_animation(image, audio_clip.duration)
        
        # Create video clip
        video_clip = ImageSequenceClip(frames, fps=30)
        
        # Set audio of the clip
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write video file
        final_clip.write_videofile(temp_video.name, codec='libx264', audio_codec='aac')
        
        return temp_video.name
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        return None

# Start listening button
if st.button("Start Listening"):
    user_input = listen_to_user()
    if user_input:
        # Generate response using the Assistant
        response = generate_response(user_input)
        if response:
            st.write(f"Generated response: {response}")

            # Convert response to speech
            audio_file = text_to_speech(response)
            if audio_file:
                st.audio(audio_file)

                # Create talking avatar video
                if generated_image is not None:
                    with st.spinner("Creating talking avatar video..."):
                        video_path = create_talking_avatar(generated_image, audio_file)
                        if video_path:
                            st.video(video_path)
                else:
                    st.warning("Please generate an image first before creating a video.")

# Add a footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")