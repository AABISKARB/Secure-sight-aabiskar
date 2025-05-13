from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify, send_from_directory
import time
from datetime import datetime
import os
from datetime import datetime
import cv2
import subprocess
import threading
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from werkzeug.utils import secure_filename
from test import predict_video  # Import the predict_video function for video file processing
from collections import deque
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client  # For sending notifications
from test import send_whatsapp_message 

# Configuration and constants
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST = ["Violence", "Normal"]  # Updated to show "Normal" for non-violent predictions
BUFFER_SIZE = 150
FRAME_SKIP = 1           # Process every frame (or adjust as needed)
PREDICTION_INTERVAL = 5  # Run prediction every 5 processed frames
last_notification_time = 0

# Load the model
model = load_model(r"D:\Aabishkar-secu\Aabishkar-secu\N_MoBiLSTM_model_final.h5")

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def save_violence_frames_rt(frames_queue, frame_id):
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'frames', frame_id)
    os.makedirs(output_dir, exist_ok=True)
    # Sample 6 frames evenly from the provided list of original frames
    indices = np.linspace(0, len(frames_queue) - 1, 6, dtype=int)
    for i, idx in enumerate(indices):
        frame = frames_queue[idx]
        # Resize to a standard display size (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(os.path.join(output_dir, f"violence_frame_{i+1}.jpg"), frame)

@app.route('/uploads/frames/<frame_id>/<filename>')
def serve_frame(frame_id, filename):
    frames_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'frames', frame_id)
    response = send_from_directory(frames_folder, filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

@app.route('/')
def Home():
    return render_template('index.html')
@app.route('/detect')
def detection():
    return render_template('detectionmethod.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    message = ""
    files = []
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'frames')
    frame_id = None  # Initialize frame_id

    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                message = 'No selected file'
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    # Save the uploaded file
                    file.save(filepath)
                    # Process the video file
                    result, frame_id = predict_video(filepath, SEQUENCE_LENGTH=32)
                    message = f"Detected Category: {result}"
                    if result == "Violence":
                        frame_folder = os.path.join(output_folder, frame_id)
                        files = os.listdir(frame_folder)
                except Exception as e:
                    message = f"Error processing video: {str(e)}"
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
    timestamp = datetime.now().timestamp()
    return render_template('upload.html', message=message, files=files, frame_id=frame_id, timestamp=timestamp)

@app.route("/frames/<frame_id>")
def show_frames(frame_id):
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'frames', frame_id)
    files = os.listdir(output_folder)
    timestamp = datetime.now().timestamp()
    return render_template('frames.html', files=files, frame_id=frame_id, timestamp=timestamp)

@app.route('/webcam')
def webcam_upload():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global last_notification_time  # so we can update the global timer
        video_capture = cv2.VideoCapture(0)  # Open the webcam
        frames_buffer = deque(maxlen=BUFFER_SIZE)             # Buffer for normalized frames (for prediction)
        frames_buffer_original = deque(maxlen=BUFFER_SIZE)      # Buffer for original frames (for display)
        frame_count = 0
        current_prediction = "Initializing..."
        text_color = (0, 255, 0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break   

            frame_count += 1
            # Append the original frame to the original frames buffer
            frames_buffer_original.append(frame.copy())

            # Process non-skipped frames as before:
            if frame_count % FRAME_SKIP != 0:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (40, 40, 43), -1)
                cv2.putText(frame, current_prediction, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                continue

            # Prepare frame for prediction:
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_buffer.append(normalized_frame)

            # When buffer is full and at defined intervals, run prediction
            if len(frames_buffer) >= BUFFER_SIZE and frame_count % PREDICTION_INTERVAL == 0:
                skip_frames_window = max(int(BUFFER_SIZE / SEQUENCE_LENGTH), 1)
                selected_frames = [frames_buffer[i] for i in range(0, BUFFER_SIZE, skip_frames_window)][:SEQUENCE_LENGTH]
                if len(selected_frames) == SEQUENCE_LENGTH:
                    predicted_probs = model.predict(np.expand_dims(selected_frames, axis=0))[0]
                    predicted_label = np.argmax(predicted_probs)
                    current_prediction = CLASSES_LIST[predicted_label]
                    text_color = (0, 0, 255) if current_prediction == "Violence" else (0, 255, 0)

                    # Check if violence was detected and ensure a 30-second gap between notifications
                    if current_prediction == "Violence":
                        current_time = time.time()
                        if current_time - last_notification_time >= 30:
                            last_notification_time = current_time
                            # Create a unique identifier for this violence segment
                            frame_id = datetime.now().strftime("%Y%m%d%H%M%S")
                            # Save sampled frames from the original buffer for display
                            save_violence_frames_rt(list(frames_buffer_original), frame_id)
                            # Send WhatsApp notification with a link to view these frames
                            send_whatsapp_message(frame_id)

            # Overlay the current prediction on the frame and yield it
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (40, 40, 43), -1)
            cv2.putText(frame, current_prediction, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        video_capture.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def save_violence_frames(original_frame, frames_queue, frame_id):
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'frames', frame_id)
    os.makedirs(output_dir, exist_ok=True)
    indices = np.linspace(0, len(frames_queue) - 1, 6, dtype=int)
    for i, idx in enumerate(indices):
        frame = (frames_queue[idx] * 255).astype(np.uint8)
        frame = cv2.resize(frame, (640, 480))
        cv2.imwrite(os.path.join(output_dir, f"violence_frame_{i+1}.jpg"), frame)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)