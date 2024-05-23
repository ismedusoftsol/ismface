import os
from datetime import datetime
import time  # Add this line to import the time module
import pyautogui as pyautogui
from flask import Flask, render_template, Response
import cv2
import math
import pyaudio
import wave
import numpy as np

# ismproctoring

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Use the default camera

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_DURATION = 10  # Duration to record when distortion is detected (in seconds)
THRESHOLD = 1000  # Example threshold for distortion detection
# Initialize audio stream
stream = None  # Will be initialized later
# Initialize PyAudio
audio = pyaudio.PyAudio()


# # Initialize audio stream
# stream = audio.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     frames_per_buffer=CHUNK,
#                     input=True)


def calculate_direction(previous_x, current_x):
    if current_x > previous_x:
        return "left"
    elif current_x < previous_x:
        return "right"
    else:
        return "none"


def start_audio_recording():
    global stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        frames_per_buffer=CHUNK,
                        input=True)
    record_audio()

def stop_audio_recording():
    global stream
    if stream is not None:
        stream.stop_stream()
        stream.close()
        global is_recording
        global audio_frames
        is_recording = False
        # Save audio to file
        audio_path = os.path.join("recordings", f"recorded_audio_.wav")
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

        print(f"Audio saved as: {audio_path}")


def record_audio():
    global is_recording
    global audio_frames

    is_recording = True
    audio_frames = []

    while is_recording:
        data = stream.read(CHUNK)
        audio_frames.append(data)



def generate_frames():
    previous_x = None
    angle_start_time = None
    face_start_time = None
    # start_audio_recording()  # Start audio recording
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Load the Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 1:
                save_photo(frame)
            if len(faces) == 1:
                # Consider only the first detected face
                x, y, w, h = faces[0]
                # Calculate the center of the face
                center_x = x + w // 2
                center_y = y + h // 2
                if previous_x is not None:
                    # Calculate the angle of movement
                    angle = math.degrees(math.atan2(center_y - y, center_x - previous_x))
                    print(angle)
                    if angle < 90:  # Check if angle is less than 90 degrees
                        if angle_start_time is None:
                            angle_start_time = datetime.now()
                        elif (datetime.now() - angle_start_time).total_seconds() >= 3:
                            # Save the photo
                            save_photo(frame)
                            angle_start_time = None
                            # Start audio recording

                    else:
                        angle_start_time = None
                previous_x = center_x
                face_start_time = None
            else:
                # Start the timer if no face is detected
                if face_start_time is None:
                    face_start_time = datetime.now()
                elif (datetime.now() - face_start_time).total_seconds() >= 2:
                    # Save the photo if no face is detected for 3 seconds
                    save_photo(frame)
                    face_start_time = None
                previous_x = None
            # Draw rectangle around the face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()  # Convert frame to bytes
            # Concatenate frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Check for distortion and trigger audio recording

    # stop_audio_recording()  # Stop audio recording when video streaming ends


@app.route('/capture_photo')
def capture_photo_route():
    capture_photo()
    return "Photo captured!"


def capture_photo():
    # Capture screenshot
    screenshot = pyautogui.screenshot()
    # Convert RGBA image to RGB
    screenshot = screenshot.convert('RGB')
    # Save the screenshot as a photo
    if not os.path.exists("screenshots_photos"):
        os.makedirs("screenshots_photos")
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    photo_path = os.path.join("screenshots_photos", f"screenshot_{timestamp}.jpg")
    screenshot.save(photo_path)
    print(f"Screenshot saved as: {photo_path}")
    # stop_audio_recording()
    return "Photo captured successfully!"
    # success, frame = camera.read()  # Read the camera frame
    # if success:
    #     save_photo(frame)


def save_photo(frame):
    # Create the directory if it doesn't exist
    if not os.path.exists("captured_photos"):
        os.makedirs("captured_photos")
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join("captured_photos", f"captured_photo_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print("Photo saved:", filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/close_application', methods=['POST'])
def close_application():
    # Trigger your method here
    # Method to be triggered when the user clicks "Yes" in the confirmation dialog
    print("Method triggered successfully!")
    return "Application closed successfully!"


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
