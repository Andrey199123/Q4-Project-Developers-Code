# BETA 1.0.0
# ALL FUNCTIONS WITHIN THIS DOCUMENT WERE WRITTEN BY DESERTRON FOR THE GROUP ROBOT MOVEMENT PROJECT
# Import necessary modules
import platform
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from datetime import datetime
import requests
import os
import subprocess
import logging
from queue import Queue
from threading import Lock
import json
from LoginPage import validate_login, register_user, create_table
from LineProcessing import stream_processing # Import frame processing from LineProcessing.py
import cv2
# Initialize Flask application and configurations
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')
RPI_IP = "192.168.240.22"  # Update with your Raspberry Pi's IP
RPI_USER = "pi"      # Update with your Raspberry Pi username
create_table()
# Global variables
message = ""
login_logs = []
command_logs = []
StartStatus = 3
event_queue = Queue()
clients = set()
video_Stream = False
clients_lock = Lock()
VIDEO_PATH = 0
# Logging configuration
LOG_FILE_PATH = "app.log"
LOGIN_LOG_FILE_PATH = "login.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# event_type - variable containing event message that needs to be sent
# data - actual information that needs to be sent
def broadcast_event(event_type, data):
    """Send event to all connected clients."""
    event = {'type': event_type, 'data': data}
    with clients_lock:
        for client_queue in clients:
            client_queue.put(event)
# username - username logging in
def log_login(username):
    """Log login attempt and broadcast event."""
    login_message = f"{username} logged in at {datetime.now()}"
    with open(LOGIN_LOG_FILE_PATH, 'a') as f:
        f.write(login_message + "\n")
    logging.info(login_message)
    broadcast_event('login_log', login_message)

def check_services():
    """Check if both scripts are running on the Raspberry Pi."""
    try:
        cmd = f"ssh {RPI_USER}@{RPI_IP} 'pgrep -f \"python3 (RPIcameraStream|rasperryPiControl).py\"'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

# action that the robot needs to take
def control_robot(action):
    """Send control commands to the robot."""
    global message
    try:
        response = requests.post(f"http://{RPI_IP}:5010/control_robot", json={'action': action})
        if response.status_code == 200:
            message = f"Command '{action}' sent successfully"
        else:
            message = f"Error sending command '{action}'"
    except Exception as e:
        message = f"Failed to connect to Raspberry Pi: {e}"

    log_entry = {
        "command": action,
        "timestamp": str(datetime.now()),
        "status": message
    }
    command_logs.append(log_entry)
    logging.info(message)
    broadcast_event('command_log', log_entry)
    return message

@app.route('/')
def login():
    """Render the login page."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_user():
    """Handle user login."""
    username = request.form['username']
    password = request.form['password']
    if validate_login(username, password):
        session['username'] = username
        log_login(username)
        return redirect(url_for('dashboard'))
    return 'Invalid credentials', 401

@app.route('/register', methods=['POST'])
def register_user_route():
    """Handle user registration."""
    username = request.form['username']
    password = request.form['password']
    if register_user(username, password):
        return redirect(url_for('login'))
    return 'Registration failed. Username may already exist.', 400

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard."""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])


@app.route('/video_feed/raw')
def raw_video_feed():
    def generate_frames():
            cap = cv2.VideoCapture(VIDEO_PATH)
            while video_Stream == True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Could not read raw frame")
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                if buffer is None:
                    print("Error: Could not encode raw frame")
                    continue

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/overlay')
def overlay_video_feed():
    def generate_frames():
            cap = cv2.VideoCapture(VIDEO_PATH)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            while video_Stream:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Could not read frame from video")
                    break

                processed_frame = stream_processing(frame)

                if processed_frame is None:
                    print("Error: Processed frame is None, using original frame")
                    processed_frame = frame  # Use the original frame if processing fails

                _, buffer = cv2.imencode('.jpg', processed_frame)
                if buffer is None:
                    print("Error: Could not encode frame")
                    continue  # Skip this frame

                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            cap.release()


    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    """SSE endpoint for real-time updates."""
    def generate():
        client_queue = Queue()
        with clients_lock:
            clients.add(client_queue)
        try:
            while True:
                message = client_queue.get()
                yield f"data: {json.dumps(message)}\n\n"
        finally:
            with clients_lock:
                clients.remove(client_queue)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/get-login-logs')
def get_login_logs():
    """Return login logs."""
    if os.path.exists(LOGIN_LOG_FILE_PATH):
        with open(LOGIN_LOG_FILE_PATH, 'r') as f:
            logs = f.readlines()
    else:
        logs = ["Login log file not found."]
    return jsonify(logs=logs)

@app.route('/get-logs')
def get_logs():
    """Return general logs."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as f:
            logs = f.readlines()
    else:
        logs = ["Log file not found."]
    return jsonify(logs=logs)


@app.route('/send-command', methods=['POST'])
def send_command():
    """Handle robot control commands."""
    if 'username' not in session:
        return redirect(url_for('login'))

    command = request.json.get('command')
    response_message = control_robot(command)

    return jsonify({"status": response_message})

@app.route('/start-robot', methods=['POST'])
def start_robot():
    """Start or stop both the camera stream and robot control scripts."""
    global StartStatus
    global video_Stream
    action = request.json.get('action')
    message = ""

    if action == "start":
        video_Stream = True

    elif action == "exit":
        video_Stream = False
        if StartStatus == 3:
            try:
                # Kill both scripts using SSH
                ssh_commands = [
                    f"ssh {RPI_USER}@{RPI_IP} 'pkill -f sudo service motion restart'",
                    f"ssh {RPI_USER}@{RPI_IP} 'pkill -f RasperryPiControl.py'"
                ]

                for cmd in ssh_commands:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception(f"Command failed: {result.stderr}")

                StartStatus = False
                message = "Robot and camera stream stopped successfully."
            except Exception as e:
                message = f"Error stopping scripts: {str(e)}"
        else:
            message = "The robot is not currently running."

    log_entry = {
        "command": action,
        "timestamp": str(datetime.now()),
        "status": message
    }
    command_logs.append(log_entry)
    logging.info(f"Command executed: {message}")
    broadcast_event('command_log', log_entry)
    return jsonify({"status": message})


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5001)

