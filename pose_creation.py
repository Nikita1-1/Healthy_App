import mediapipe as mp
import numpy as np
import math
import cv2



class CreatePose_Front():
    def __init__(self):
       pass

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

    def leg_alignment_status(self, midpoint_hip_x, left_knee, right_knee, threshold=0.003):
        # Calculate distances
        distance_to_left_knee = math.sqrt(
            (midpoint_hip_x[0] - left_knee[0]) ** 2 + (midpoint_hip_x[1] - left_knee[1]) ** 2)
        distance_to_right_knee = math.sqrt(
            (midpoint_hip_x[0] - right_knee[0]) ** 2 + (midpoint_hip_x[1] - right_knee[1]) ** 2)

        # Calculate the difference in distances
        distance_diff = abs(distance_to_left_knee - distance_to_right_knee)

        # Calculate the deviation from normal alignment
        deviation = distance_to_left_knee - distance_to_right_knee

        # Compare with threshold to determine alignment status
        if distance_diff > threshold:
            if deviation > 0:
                return f"Left leg is not aligned, Deviation from normal: {abs(deviation)}"
            else:
                return f"Right leg is not aligned, Deviation from normal: {abs(deviation)}"
        else:
            return "Both legs are aligned properly."


class CreatePose_Profile():
    def __int__(self):
        pass

    def shoulder_alignment_profile(self, left_shoulder, right_shoulder, threshold=0.1):
        # Calculate the difference in x-coordinates between left and right shoulders
        shoulder_diff_x = left_shoulder[0] - right_shoulder[0]

        # Check for forward inclination
        if shoulder_diff_x > threshold:
            forward_amount = shoulder_diff_x - threshold
            return f"Left shoulder is moved forward by {forward_amount:.2f} units."

        # Check for backward inclination
        elif shoulder_diff_x < -threshold:
            backward_amount = abs(shoulder_diff_x) - threshold
            return f"Right shoulder is moved forward by {backward_amount:.2f} units."

        else:
            return "Shoulders alignment is fine."


    def foot_index_alignment(self, left_foot_index, right_foot_index, threshold = 0.008):
        foot_index_different_x = left_foot_index[0] - right_foot_index[0]

        if foot_index_different_x > threshold:
            forward_amount = foot_index_different_x - threshold
            return f"Left foot is moved forward by {forward_amount:.2f} units."

            # Check for backward inclination
        elif foot_index_different_x < -threshold:
            backward_amount = abs(foot_index_different_x) - threshold
            return f"Right foot is moved forward by {backward_amount:.2f} units."