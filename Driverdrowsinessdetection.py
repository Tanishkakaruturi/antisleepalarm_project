import serial
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import numpy as np
from cv2 import cv2
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
# Initialize Arduino communication
arduino = serial.Serial('COM7', 9600)
time.sleep(2)

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Initialize the video stream
print("[INFO] initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Frame dimensions
frame_width = 1024
frame_height = 576

# Get indices for facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Constants
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.74

# Timing variables
eye_closed_time = 0  # Timer for eyes closed
yawning_counter = 0   # Counter for yawns
yawning_time = 0      # Timer for yawning

# Grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# Initialize image points for head tilt calculation
image_points = np.zeros((6, 2), dtype='double')

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=frame_width, height=frame_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)

    # Check for detected faces
    if len(rects) > 0:
        for rect in rects:
            # Get the bounding box for the face
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Compute eye aspect ratio
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if eyes are closed
            if ear < EYE_AR_THRESH:
                eye_closed_time += 1  # Increment timer for eyes closed
                if eye_closed_time >= 40:  # If eyes closed for more than 4 seconds
                    arduino.write(b'H')  # Send signal to Arduino
                    cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                eye_closed_time = 0  # Reset timer when eyes are open
                arduino.write(b'N')   # Normal condition signal

            # Compute mouth aspect ratio
            mouth = shape[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)

            # Draw mouth contours
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mouthMAR), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if yawning
            if mouthMAR > MOUTH_AR_THRESH:
                yawning_time += 1  # Increment yawning timer
                if yawning_time >= 30:  # If yawning for more than 3 seconds
                    yawning_counter += 1  # Increment yawn count
                    if yawning_counter >= 2:  # If yawning 3 times
                        arduino.write(b'H')  # Send signal to Arduino
                        cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawning_time = 0  # Reset timer for yawning
                yawning_counter = 0 if yawning_counter > 0 else yawning_counter  # Reset counter if mouth is not open

            # Draw facial landmarks
            for (i, (x, y)) in enumerate(shape):
                # Highlight specific landmarks
                if i == 33:
                    image_points[0] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    image_points[1] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                    image_points[2] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                    image_points[3] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    image_points[4] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                    image_points[5] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # Draw the determinant image points onto the person's face
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # Calculate head tilt and draw lines
            (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords((frame_width, frame_height), image_points, frame_height)
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            if head_tilt_degree:
                cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
