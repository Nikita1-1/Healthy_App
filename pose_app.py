import mediapipe as mp
import numpy as np
import cv2
import math
from video_improve import reduce_noise, draw_grid, draw_square
from exctracting_keypoints import Draw_Profile_View
from process_landmarks_for_pose import ProcessLandmarks

import time as tm
import os

class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def pose_front_view(self):
        print("Please, follow the image on the screen and take a pose!")

    def main_function(self):
        cap = cv2.VideoCapture(0)
        with self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8,
                               smooth_landmarks=True) as pose:
            while True:  # Keep the loop running indefinitely
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from camera.")
                    break  # Exit the loop if there's an error

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                results = pose.process(image)

                try:
                    denoised_image = reduce_noise(image)
                    exctractin_keyp = Draw_Profile_View()
                    exctractin_keyp.draw_custom_skeleton(denoised_image, results, image_width, image_height,
                                                         (255, 255, 255))
                    draw_grid(denoised_image, rows=22, cols=22, color=(128, 128, 128), thickness=1)
                    draw_square(denoised_image, (11, 11), (625, 464), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error: {e}")
                    pass

                image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                           circle_radius=2),
                                               self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                           circle_radius=2))
                cv2.imshow('Pose Detection', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit the loop if 'q' is pressed

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main_function()


