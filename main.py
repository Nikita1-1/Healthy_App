from pose_app import PoseDetector
from create_folders import Folder_Teach_ML
if __name__ == "__main__":
    detector = PoseDetector()
    detector.main_function()
    folder_teach_ml = Folder_Teach_ML()
    folder_teach_ml.folder_for_ml()