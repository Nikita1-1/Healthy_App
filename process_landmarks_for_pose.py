
from pose_creation import CreatePose_Front
class ProcessLandmarks:
    def __init__(self):
        pass


    def process_for_front_view(self, landmarks):
        left_shoulder = landmarks["shoulder"]
        right_shoulder = landmarks["shoulder_r"]
        left_elbow = landmarks["elbow"]
        right_elbow = landmarks["elbow_r"]
        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]
        neck_point_x = landmarks["neck_point_x"]
        neck_point_y = landmarks["neck_point_y"]
        midpoint_hip_x = landmarks["midpoint_hip_x"]
        midpoint_hip_y = landmarks["midpoint_hip_y"]
        poses = CreatePose_Front()
        poses.hips_alignment(left_hip, right_hip, threshold=0.004)
        poses.shoulder_alignment(left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip,
                           threshold=0.003)
        poses.spine_alignment(neck_point_x, neck_point_y, midpoint_hip_x, midpoint_hip_y, 0.003)