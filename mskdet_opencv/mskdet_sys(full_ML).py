# import dependencies
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# define a function to detect and predict the mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    # dimensions of the frame and construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize the list of faces, locations, and predictions
    faces = []
    face_locs = []
    mask_preds = []

    # loop over the detections and filter out weak detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding face_box for the object
            face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X_start, y_start, X_end, y_end) = face_box.astype("int")
            (X_start, y_start) = (max(0, X_start), max(0, y_start))
            (X_end, y_end) = (min(w - 1, X_end), min(h - 1, y_end))
            face = frame[y_start:y_end, X_start:X_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            face_locs.append((X_start, y_start, X_end, y_end))

        else:
            print("No face detected")
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        mask_preds = maskNet.predict(faces, batch_size=32)
    return (face_locs, mask_preds),

# face detector model (opensourced model from OpenCV)
prototxtPath = r"C:\Users\Chamod Peiris\Documents\GitHub\Projects_24\Sys_MaskDetection\face_detector\deploy.prototxt"
weightsPath = r"C:\Users\Chamod Peiris\Documents\GitHub\Projects_24\Sys_MaskDetection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# mask detector model (trained model from Face-Mask-Detection)
maskNet = load_model(r"C:/Users/Chamod Peiris/Documents/GitHub/Projects_24/Sys_MaskDetection/mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # resize the frame dimensions
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # face detection and mask prediction
    (face_locs, mask_preds), = detect_and_predict_mask(frame, faceNet, maskNet)


    for (face_box, pred) in zip(face_locs, mask_preds):
        # unpack the bounding face_box and predictions
        (X_start, y_start, X_end, y_end) = face_box
        (mask, withoutMask) = pred

        # detection label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # draw the bounding face_box and text
        cv2.putText(frame, label, (X_start, y_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (X_start, y_start), (X_end, y_end), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # break the frame
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()