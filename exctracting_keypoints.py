import numpy as np
import cv2
import mediapipe as mp
from process_landmarks_for_pose import ProcessLandmarks
class Draw_Profile_View:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_custom_skeleton(self, image, results, image_width, image_height, line_color):
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if all(landmarks):
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist =  [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]

                # Get Tje Corridnates of Hip
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_heel =  [landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                right_heel = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0
                # fOR eAR
                landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

                # Check if both shoulders are visible.
                right_foot_index = [landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                left_foot_index = [landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                midpoint_shoulder_x = (int(shoulder[0] * image_width) + int(shoulder_r[0] * image_width)) / 2

                midpoint_shoulder_y = (int(shoulder[1] * image_height) + int(shoulder_r[1] * image_height)) / 2

                midpoint_hip_x = (int(left_hip[0] * image_width) + int(right_hip[0] * image_width)) / 2
                midpoint_hip_y = (int(left_hip[1] * image_height) + int(right_hip[1] * image_height)) / 2

                based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x) / 2)
                based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y) / 2)

                neck_point_x = (int(nose[0] * image_width) + int(midpoint_shoulder_x)) / 2
                neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y)) / 2

                mid_point_x = (int(left_hip[0] * image_width) + int(right_hip[0] * image_width)) / 2
                mid_point_y = (int(left_hip[1] * image_height) + int(right_hip[1] * image_height)) / 2

                point_between_neck_hip_x = int((neck_point_x + mid_point_x) / 2)
                point_between_neck_hip_y = int((neck_point_y + mid_point_y) / 2)

                point_between_neck_and_mid_x = int((neck_point_x + point_between_neck_hip_x) / 2)
                point_between_neck_and_mid_y = int((neck_point_y + point_between_neck_hip_y) / 2)

                point_between_mid_and_hip_x = int((point_between_neck_hip_x + mid_point_x) / 2)
                point_between_mid_and_hip_y = int((point_between_neck_hip_y + mid_point_y) / 2)



                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0


                # Draw lines connecting landmarks
                cv2.line(image, (int(nose[0] * image_width), int(nose[1] * image_height)),
                         (int(neck_point_x), int(neck_point_y)), line_color, 2)
                cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),
                         (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), line_color, 2)
                cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),
                         (int(neck_point_x), int(neck_point_y)), line_color, 2)
                cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)),
                         (int(neck_point_x), int(neck_point_y)), line_color, 2)
                cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),
                         (int(elbow[0] * image_width), int(elbow[1] * image_height)), line_color, 2)
                cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)),
                         (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), line_color, 2)
                cv2.line(image, (int(elbow[0] * image_width), int(elbow[1] * image_height)),
                         (int(left_wrist[0] * image_width), int(left_wrist[1] * image_height)), line_color, 2)
                cv2.line(image, (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)),
                         (int(right_wrist[0] * image_width), int(right_wrist[1] * image_height)), line_color, 2)

                #drawing mid line body
                cv2.line(image, (int(neck_point_x), int(neck_point_y)), (int(point_between_neck_and_mid_x), int(point_between_neck_and_mid_y)), line_color, 2,
                         cv2.LINE_4)
                cv2.line(image, (int(point_between_neck_and_mid_x), int(point_between_neck_and_mid_y)), (int(based_mid_x), int(based_mid_y)), line_color, 2,
                         cv2.LINE_8)
                cv2.line(image, (int(based_mid_x), int(based_mid_y)), (int(point_between_mid_and_hip_x), int( point_between_mid_and_hip_y)), line_color, 2,
                         cv2.LINE_8)
                cv2.line(image, (int(point_between_mid_and_hip_x), int(point_between_mid_and_hip_y)), (int(midpoint_hip_x), int(midpoint_hip_y)), line_color, 2,
                         cv2.LINE_8)


                #upper body lines
                cv2.line(image, (0, int(midpoint_shoulder_y)), (image.shape[1], int(midpoint_shoulder_y)), (0, 1, 255), 1,
                         cv2.LINE_8)
                cv2.line(image, (0, int(midpoint_hip_y)), (image.shape[1], int(midpoint_hip_y)), (3, 0, 255), 1, cv2.LINE_8)
                cv2.line(image, (int(right_hip[0] * image_width), int(right_hip[1] * image_height)),
                         (int(right_knee[0] * image_width), int(right_knee[1] * image_height)), line_color,2)
                cv2.line(image, (int(left_hip[0] * image_width), int(left_hip[1] * image_height)),
                         (int(left_knee[0] * image_width), int(left_knee[1] * image_height)), line_color, 2)
                cv2.line(image, (int(right_knee[0] * image_width), int(right_knee[1] * image_height)),
                         (int(right_ankle[0] * image_width), int(right_ankle[1] * image_height)), line_color, 2)
                cv2.line(image, (int(left_knee[0] * image_width), int(left_knee[1] * image_height)),
                         (int(left_ankle[0] * image_width), int(left_ankle[1] * image_height)), line_color, 2)
                cv2.line(image, (int(left_ankle[0] * image_width), int(left_ankle[1] * image_height)),
                         (int(left_heel[0] * image_width), int(left_heel[1] * image_height)), line_color, 2)
                cv2.line(image, (int(right_ankle[0] * image_width), int(right_ankle[1] * image_height)),
                         (int(right_heel[0] * image_width), int(right_heel[1] * image_height)), line_color, 2)
                cv2.line(image, (int(left_ankle[0] * image_width), int(left_ankle[1] * image_height)),
                         (int(left_foot_index[0] * image_width), int(left_foot_index[1] * image_height)), line_color, 2)

                cv2.line(image, (int(right_ankle[0] * image_width), int(right_ankle[1] * image_height)),
                         (int(right_foot_index[0] * image_width), int(right_foot_index[1] * image_height)), line_color, 2)




                # Draw circles at specific points
                circle_color = (255, 191, 0)
                cv2.circle(image, (int(midpoint_hip_x), int(midpoint_hip_y)), 2, circle_color, -1)
                cv2.circle(image, (int(mid_point_x), int(mid_point_y + 30)), 10, circle_color, -1)
                cv2.circle(image, (int(neck_point_x), int(neck_point_y)), 2, circle_color, 5)
                cv2.circle(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 2, circle_color, 3)
                cv2.circle(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 2, circle_color, 3)
                cv2.circle(image, (int(based_mid_x), int(based_mid_y)), 2, circle_color, 5)
                cv2.circle(image, (int(point_between_neck_hip_x), int(point_between_neck_hip_y)), 2, circle_color, 5)
                cv2.circle(image, (int(point_between_neck_and_mid_x), int(point_between_neck_and_mid_y)), 2, circle_color, 5)
                cv2.circle(image, (int(point_between_mid_and_hip_x), int(point_between_mid_and_hip_y)), 2, circle_color, 5)
                cv2.circle(image, (int(neck_point_x - 30), int(neck_point_y + 30)), 8, circle_color, 2)

                landmarks_dict = {"shoulder": shoulder, "shoulder_r":shoulder_r, "left_elbow":elbow, "right_elbow":elbow_r, "left_hip": left_hip, "right_hip":right_hip, "neck_point_x":neck_point_x, "neck_point_y":neck_point_y, "midpoint_hip_x":midpoint_hip_x, "midpoint_hip_y":midpoint_hip_y,
                                  "left_knee":left_knee, "right_knee":right_knee, "left_foot_index":left_foot_index, "right_foot_index":right_foot_index}

                create_pose_front = ProcessLandmarks()
                create_pose_front.process_for_front_view(landmarks_dict)
                create_pose_front.process_for_profile_view(landmarks_dict)
                return landmarks

        return None
