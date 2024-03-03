
from pose_creation import *
class ProcessLandmarks:
    def __init__(self):
        pass


    def process_for_front_view(self, landmarks):
        left_shoulder = landmarks["shoulder"]
        right_shoulder = landmarks["shoulder_r"]
        left_elbow = landmarks["left_elbow"]
        right_elbow = landmarks["right_elbow"]
        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]
        neck_point_x = landmarks["neck_point_x"]
        neck_point_y = landmarks["neck_point_y"]
        midpoint_hip_x = landmarks["midpoint_hip_x"]
        midpoint_hip_y = landmarks["midpoint_hip_y"]
        left_knee = landmarks["left_knee"]
        right_knee = landmarks["right_knee"]
        poses = CreatePose_Front()
        poses.hips_alignment(left_hip, right_hip, 0.004)
        poses.shoulder_alignment(left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip,0.003)
        poses.spine_alignment(neck_point_x, neck_point_y, midpoint_hip_x, midpoint_hip_y, 0.003)
        poses.angle_between_hip_leg(midpoint_hip_y, midpoint_hip_x, left_knee, right_knee, 0.003)


    def process_for_profile_view(self, landmarks):
        left_shoulder = landmarks["shoulder"]
        right_shoulder = landmarks["shoulder_r"]
        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]
        left_knee = landmarks["left_knee"]
        right_knee = landmarks["right_knee"]
        left_foot_index = landmarks["left_foot_index"]
        right_foot_index = landmarks["right_foot_index"]
        poses_profile = CreatePose_Profile()
        poses_profile.shoulder_alignment_profile(left_shoulder, right_shoulder, 0.008)
        poses_profile.foot_index_alignment(left_foot_index, right_foot_index, 0.008)