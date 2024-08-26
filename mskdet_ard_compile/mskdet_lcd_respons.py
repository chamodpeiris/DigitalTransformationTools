import cv2
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import serial
import time

# Load the MediaPipe face detection and pose detection models
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
pose_detection = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the mask detection model
maskNet = load_model("C:/Users/Chamod Peiris/Documents/GitHub/Projects_24/Sys_MaskDetection/mask_detector.model")

# Initialize the video stream
vs = VideoStream(src=0).start()

# Initialize serial communication with Arduino
ser = serial.Serial('COM5', 9600)  # Replace 'COM3' with the appropriate serial port

# Variable to keep track of mask detection state
prev_state = None

# Loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    frame = cv2.resize(frame, (1200, 800))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MediaPipe
    results = face_detection.process(frame_rgb)
    if results.detections:
        for detection in results.detections:
            ih, iw, _ = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            (startX, startY, w, h) = bbox
            endX = startX + w
            endY = startY + h

            # Extract face ROI
            face = frame[startY:endY, startX:endX]

            # Check if face is not empty before converting color
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert color
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Predict mask
            (mask, withoutMask) = maskNet.predict(face)[0]

            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask" and prev_state != 'M':
                ser.write(b'M')  # Send 'M' to Arduino for mask detected
                prev_state = 'M'
            elif label == "No Mask" and prev_state != 'N':
                ser.write(b'N')  # Send 'N' to Arduino for no mask detected
                prev_state = 'N'

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Display face coordinates
            cv2.putText(frame, f"X: {startX}, Y: {startY}", (startX, startY - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Detect body pose using MediaPipe
    pose_results = pose_detection.process(frame_rgb)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()