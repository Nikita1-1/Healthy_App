import mediapipe as mp
import numpy as np
import cv2

from pose_app import PoseDetector
from exctracting_keypoints import Draw_Profile_View

class CreatePose_Front(PoseDetector, Draw_Profile_View):
    def __init__(self):
        super().__init__()

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def shoulder_alignment(self, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip,
                           threshold=0.003):
        left_shoulder_y, right_shoulder_y = left_shoulder[1], right_shoulder[1]
        left_elbow_y, right_elbow_y = left_elbow[1], right_elbow[1]

        # Check if the difference in y-coordinates between shoulders and elbows is within the threshold
        shoulder_elbow_diff = abs(left_shoulder_y - left_elbow_y) - abs(right_shoulder_y - right_elbow_y)

        # Calculate the difference in distance between shoulders and hips
        shoulder_hip_diff = np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)) - np.linalg.norm(
            np.array(right_shoulder) - np.array(right_hip))

        # Determine which side is misaligned
        higher_side = None
        lower_side = None
        if shoulder_elbow_diff > threshold:
            higher_side = "left"
            lower_side = "right"
        elif shoulder_elbow_diff < -threshold:
            higher_side = "right"
            lower_side = "left"

        # Check alignment based on the factors
        if abs(shoulder_elbow_diff) > threshold or abs(shoulder_hip_diff) > threshold:
            if higher_side and lower_side:
                return f"The {higher_side} side is higher and the {lower_side} side is lower."
            else:
                return "Both sides are misaligned."
        else:
            return "Aligned"

    def hips_alignment(self, left_hip, right_hip, threshold=0.009):
        left_hip_y = left_hip[1]
        right_hip_y = right_hip[1]
        difference = abs(left_hip_y - right_hip_y)

        if difference < threshold:
            return "hips are aligned!"
        elif left_hip_y > right_hip_y:
            return "Left hip is lower!"
        else:
            return "Right hip is lower!"

    def spine_alignment(self, neck_point_x, neck_point_y, midpoint_hip_x, midpoint_hip_y, threshold=0.003):
        if abs(neck_point_y - midpoint_hip_y) > threshold:
            return "Not vertically aligned"
        else:
            # Determine which side is more to the left or right
            if neck_point_x > midpoint_hip_x:
                return "Neck point is more to the right side."
            elif neck_point_x < midpoint_hip_x:
                return "Neck point is more to the left side."
            else:
                return "Neck point and hip point are aligned horizontally."