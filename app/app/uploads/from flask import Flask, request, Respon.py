from flask import Flask, request, Response, render_template_string, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exercise_type_global = "push-up"
video_path_global = None


def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]


def score_table(exercise, frame, counter, status):
    cv2.putText(frame, "Activity: " + exercise.replace("-", " "),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Counter: " + str(counter), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status: " + str(status), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


@app.route('/', methods=['GET', 'POST'])
def index():
    global exercise_type_global, video_path_global

    if request.method == 'POST':
        # Get exercise type from the form
        exercise_type_global = request.form.get('exercise_type')

        # Handle video upload
        if 'video_file' not in request.files:
            return "No video file uploaded!"
        video_file = request.files['video_file']
        if video_file.filename == '':
            return "No selected file!"
        if video_file:
            video_path_global = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path_global)
            return redirect(url_for('video_feed'))

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exercise Tracker</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                text-align: center;
                margin-top: 50px;
            }
            h1 {
                color: #333;
                font-size: 2.5rem;
            }
            form {
                margin-bottom: 20px;
            }
            .form-group {
                margin: 15px 0;
            }
            label {
                font-size: 1.2rem;
                color: #555;
            }
            select, input {
                padding: 10px;
                font-size: 1rem;
                margin-top: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                padding: 10px 20px;
                font-size: 1.2rem;
                background-color: #5cb85c;
                color: #fff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #4cae4c;
            }
            .video-container {
                margin: 20px auto;
                width: 800px;
                height: 480px;
                border: 5px solid #555;
                border-radius: 15px;
                overflow: hidden;
            }
            img {
                width: 100%;
                height: auto;
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Exercise Tracker</h1>
            <form action="/" method="POST" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="exercise_type">Select Exercise:</label>
                    <select name="exercise_type" id="exercise_type" required>
                        <option value="push-up">Push-Up</option>
                        <option value="pull-up">Pull-Up</option>
                        <option value="squat">Squat</option>
                        <option value="sit-up">Sit-Up</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="video_file">Upload Video File:</label>
                    <input type="file" name="video_file" id="video_file" accept="video/*" required>
                </div>
                <button type="submit">Submit</button>
            </form>
            <div class="video-container">
                <h2>Processed Video:</h2>
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
        </div>
    </body>
    </html>
    ''')


def generate_frames():
    global video_path_global, exercise_type_global

    if not video_path_global:
        return

    cap = cv2.VideoCapture(video_path_global)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path_global}")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        counter = 0
        status = "up"  # Start in the "up" position
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                # Get landmarks
                landmarks = results.pose_landmarks.landmark

                # Calculate angles for push-up
                left_elbow_angle = calculate_angle(
                    detection_body_part(landmarks, "LEFT_SHOULDER"),
                    detection_body_part(landmarks, "LEFT_ELBOW"),
                    detection_body_part(landmarks, "LEFT_WRIST"),
                )
                right_elbow_angle = calculate_angle(
                    detection_body_part(landmarks, "RIGHT_SHOULDER"),
                    detection_body_part(landmarks, "RIGHT_ELBOW"),
                    detection_body_part(landmarks, "RIGHT_WRIST"),
                )

                # Push-up logic: Check angles to determine status
                if left_elbow_angle > 160 and right_elbow_angle > 160:  # Arms extended
                    if status == "down":
                        counter += 1  # Increment counter
                        status = "up"
                elif left_elbow_angle < 90 and right_elbow_angle < 90:  # Arms bent
                    if status == "up":
                        status = "down"

            except AttributeError:
                pass

            # Overlay information
            frame = score_table(exercise_type_global, frame, counter, status)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
