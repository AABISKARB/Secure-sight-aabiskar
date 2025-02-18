from flask import Flask, render_template, request, redirect, url_for, session, Response
import os
import cv2
import subprocess
import threading
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = 'client_secrets.json'

def upload_to_drive(file_path, file_name):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    service = build('drive', 'v3', credentials=credentials)
    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path, mimetype='video/mp4')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store the camera URL
camera_url = ""

# Route for file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    message = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part'
        file = request.files['file']
        if file.filename == '':
            message = 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            upload_to_drive(filepath, filename)
            message = 'File uploaded successfully to Google Drive'
    return render_template('upload.html', message=message)

# CCTV Streaming with OpenCV
def capture_cctv(camera_ip):
    global camera_url
    camera_url = f"rtsp://{camera_ip}"  # Update camera URL with the provided IP
    
    cap = cv2.VideoCapture(camera_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # You can add additional processing or encoding here if needed.
        # For now, we'll just show the frame.
        cv2.imshow('CCTV Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/cctv', methods=['GET', 'POST'])
def start_cctv():
    if request.method == 'POST':
        camera_ip = request.form['camera_ip']
        threading.Thread(target=capture_cctv, args=(camera_ip,), daemon=True).start()
        return f"CCTV stream started at {camera_ip}. Close the OpenCV window to stop."
    return render_template('upload.html')

# MJPEG stream (alternative way to stream CCTV via Flask)
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(camera_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Encoding the frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def execute_colab_notebook():
    subprocess.run(["gcloud", "functions", "call", "execute_notebook", "--data", '{"notebook_id": "1YVtY81WWKjViVLJOhpfdDveXDIavvf6q"}'])

if __name__ == '__main__':
    app.run(debug=True)
