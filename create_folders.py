import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

from video_improve import reduce_noise
from exctracting_keypoints import Draw_Profile_View
from pose_app import PoseDetector
class Folder_Teach_ML():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def folder_for_ml(self):
        script_directory = os.path.dirname(__file__)
        DATA_PATH = os.path.join(script_directory, "MP_Data")  # Path for exported data
        actions = np.array(["humpback neck"])# "left hip forward", "right hip forward"
        no_sequences = 23
        sequence_length = 1
        for action in actions:
            for sequence in range(no_sequences):
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

        self.read_images_for_ml(actions, no_sequences, sequence_length, DATA_PATH)

    def read_images_for_ml(self, actions, no_sequences, sequence_length, DATA_PATH):
        mp_pose = mp.solutions.pose
        # Initialize MediaPipe Holistic model
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, smooth_landmarks=True,
                          model_complexity=2) as pose:
            for action in actions:
                for sequence in range(no_sequences):
                    for frame_num in range(sequence_length):
                        image_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + ".jpg")
                        image = cv2.imread(image_path)

                        if image is None:
                            print(f"Error: Unable to read image at path {image_path}")
                            continue

                        target_size = (750, 980)
                        image = cv2.resize(image, target_size)

                        image_height, image_width, _ = image.shape
                        pose_det = Draw_Profile_View()
                        denoised_image = reduce_noise(image)
                        annotated_image = denoised_image.copy()
                        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


                        landmarks = pose_det.draw_custom_skeleton(annotated_image, results, image_width,
                                                                          image_height,
                                                                          line_color=(0, 255, 0))
                        if landmarks is not None:
                            # Extract numerical values from keypoints
                            keypoints_array = []
                            for landmark in landmarks:
                                keypoints_array.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                            keypoints_array = np.array(keypoints_array)

                            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + ".npy")
                            np.save(npy_path, keypoints_array)
                            print("Keypoints", keypoints_array)



        # Process all the data once
        x_train, x_test, y_train, y_test = self.process_data(actions, no_sequences, DATA_PATH, sequence_length)

        # Train the model using the processed data
        self.build_train_lstm(actions, x_train, y_train, x_test, y_test)

        cv2.destroyAllWindows()

    def process_data(self, actions, no_sequences, DATA_PATH, sequence_length):
        label_map = {label: num for num, label in enumerate(actions)}
        sequences, labels = [], []
        feature_dimension = 132 # Specify the dimensionality of your features

        for action in actions:
            for sequence in range(no_sequences):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        x = np.array(sequences)
        y = to_categorical(labels).astype(int)
        num_elements = len(x)
        print("Number of elements in x:", num_elements)

        print("Shape of x before reshape:", x.shape)
        print("Size of x before reshape:", x.size)
        print("Values of x before reshape:", x)
        # Reshape the input data to 3D
        x = x.reshape(x.shape[0], sequence_length, feature_dimension)

        print("Shape of x after reshape:", x.shape)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
        return x_train, x_test, y_train, y_test
    def build_train_lstm(self, actions, x_train, y_train, x_test, y_test):
        log_dir = os.path.join("Logs")
        try:
            os.makedirs(log_dir, exist_ok=True)  # Create the "Logs" directory if it doesn't exist
        except OSError as e:
            return f"Error creating 'Logs' directory: {e}"

        tb_callback = TensorBoard(log_dir=log_dir)

        num_classes = len(actions)
        input_shape = x_train.shape[1:]

        model = Sequential([
            LSTM(units=128, input_shape=input_shape),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2000, batch_size=32, callbacks=[tb_callback])

        loss, accuracy = model.evaluate(x_test, y_test)
        return f'Test Loss: {loss}, Test Accuracy: {accuracy}'


    def check_humpback_neck(self, actions):
        action = actions[0]


if __name__ == "__main__":
    detector = PoseDetector()
    detector.main_function()