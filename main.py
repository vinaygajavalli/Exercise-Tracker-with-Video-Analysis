import cv2
import mediapipe as mp
from utils import score_table
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise

def main():
    # Initialize mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Fixed video source path
    video_path = r"E:\B Tech\Hacks\HWealth\HealthIsWealth\Exercise_videos\sit-up.mp4"

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    cap.set(3, 800)  # Width
    cap.set(4, 480)  # Height

    # Setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        counter = 0  # Movement of exercise
        status = True  # State of move
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame. Exiting...")
                break

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    "sit-up", counter, status)  # Hardcoding the exercise type to "squat"
            except AttributeError:
                pass

            frame = score_table("sit-up", frame, counter, status)  # Hardcoding the exercise type to "squat"

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

            cv2.imshow('Video', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
