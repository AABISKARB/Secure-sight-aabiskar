import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client  # For sending WhatsApp messages
from datetime import datetime

# Load your pre-trained model
model = load_model(r"D:\Aabishkar-secu\Aabishkar-secu\N_MoBiLSTM_model_final.h5")
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST = ["Violence", "NonViolence"]


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio WhatsApp Sandbox number
YOUR_WHATSAPP_NUMBER = "whatsapp:+9779869670818"   # Replace with your WhatsApp number

WEB_INTERFACE_URL = "http://192.168.18.5:5000"

def send_whatsapp_message(frame_id):
    """Send a WhatsApp message with the URL of the detected violence."""
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        message_body = (
            "Violence detected! View the frames here:\n"
            f"{WEB_INTERFACE_URL}/frames/{frame_id}"
        )
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=YOUR_WHATSAPP_NUMBER,
        )
        print(f"WhatsApp message sent! SID: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return None


def predict_video_chunk(video_reader, start_frame, total_frames, SEQUENCE_LENGTH=16):
    frames_for_prediction = []
    frames_for_display = []  # This will store the original frames for display
    skip_frames_window = max(int(total_frames / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        
        # Save the original frame for display
        original_frame = frame.copy()
        
        # Resize a copy for prediction
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        
        frames_for_prediction.append(normalized_frame)
        frames_for_display.append(original_frame)
    
    if len(frames_for_prediction) < SEQUENCE_LENGTH:
        return ("Normal", None)  # Not enough frames to make a prediction
    
    predicted_probabilities = model.predict(np.expand_dims(frames_for_prediction, axis=0))[0]
    predicted_label = np.argmax(predicted_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    if predicted_class_name == "Violence":
        # Sample 6 frames evenly from the original (display) frames
        indices = np.linspace(0, len(frames_for_display) - 1, 6, dtype=int)
        sample_frames = [frames_for_display[i] for i in indices]
        return ("Violence", sample_frames)
    else:
        return ("Normal", None)

def predict_video(video_file_path, SEQUENCE_LENGTH=16):
    video_reader = cv2.VideoCapture(video_file_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the segment length (using up to 150 frames per segment)
    frame_step = min(video_frames_count, 150)

    output_folder = os.path.join(os.path.dirname(video_file_path), "frames")
    os.makedirs(output_folder, exist_ok=True)

    for start_frame in range(0, video_frames_count, frame_step):
        result, sample_frames = predict_video_chunk(video_reader, start_frame, frame_step, SEQUENCE_LENGTH)
        if result == "Violence":
            print("Predicted: Violence")
            
            # Generate a unique frame_id (e.g., timestamp)
            frame_id = datetime.now().strftime("%Y%m%d%H%M%S")
            frame_folder = os.path.join(output_folder, frame_id)
            os.makedirs(frame_folder, exist_ok=True)
            
            # Save the 6 sampled original frames
            for i, frame in enumerate(sample_frames):
                # Convert BGR to RGB for proper saving
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_path = os.path.join(frame_folder, f"violence_frame_{i+1}.png")  # Change to .jpg for JPEG format
                # Save the frame as an image file
                cv2.imwrite(output_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            # Send WhatsApp notification
            send_whatsapp_message(frame_id)
            
            video_reader.release()
            return "Violence", frame_id
    
    print("Predicted: Normal")
    video_reader.release()
    return "Normal", None