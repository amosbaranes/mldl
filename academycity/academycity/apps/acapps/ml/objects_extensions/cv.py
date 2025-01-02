import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from datetime import datetime
import pickle

from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug
#
from ....core.utils import log_debug, clear_log_debug
#
import cv2
import mediapipe as mp
import requests
#

# --------------------------

class CVAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(CVAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)

        self.app = dic["app"]


class CVDataProcessing(BaseDataProcessing, BasePotentialAlgo, CVAlgo):
    def __init__(self, dic):
        # print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.PATH = os.path.join(self.TO_OTHER, "cv")
        os.makedirs(self.PATH, exist_ok=True)
        # print(f'{self.PATH}')
        self.model = None
        self.lose_list = None
        clear_log_debug()
        #
    def run(self, dic):
        print("90755-11-cv: \n", "="*50, "\n", dic, "\n", "="*50)


        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # Open webcam or use a video file
        input_image = "pose_1.jpeg"
        output_image = "opose_1.jpeg"
        image_path = self.PROJECT_MEDIA_DIR+"/images/" + input_image
        output_path = self.PROJECT_MEDIA_DIR+"/images/" + output_image
        # print(image_path)

        image = cv2.imread(image_path)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_rgb)

            # Draw pose landmarks on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

        cv2.imwrite(output_path, image)

        # Display the image
        # cv2.imshow('Pose Estimation', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        result = {"status": "ok cv", "data":{"img":output_image}}

        return result

    def runv(self, dic):
        print("90755-11-cv-runv: \n", "="*50, "\n", dic, "\n", "="*50)

        # URL of the video
        video_url = "https://media.istockphoto.com/id/1856343710/video/yoga-fitness-and-woman-at-a-beach-with-warrior-praying-hands-or-stretching-for-mental-health.mp4?s=mp4-640x640-is&k=20&c=FzaZBijyFqjq9nrIuRzXVNXKT5NbUXf5gnHsF3g6NLE="

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://pixabay.com/videos/search/yoga/"
        }
        response = requests.get(video_url, headers=headers, stream=True)

        # Local path to save the video
        # output_path = "downloaded_video.mp4"

        input_video = "downloaded_video.mp4"
        output_video = "odownloaded_video.mp4"
        video_path = self.PROJECT_MEDIA_DIR + "/images/" + input_video  # Replace with your video file path
        output_path = self.PROJECT_MEDIA_DIR + "/images/" + output_video  # Output video file

        print("output_path:", output_path)

        # Check if the request was successful

        print("A8", response.status_code)
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Video downloaded successfully: {video_path}")
        else:
            print(f"Failed to download video. Status code: {response.status_code}")

        # Initialize MediaPipe Pose and Drawing utilities
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # Video file path (update with your video path or use the provided link)
        # input_video = "https://www.istockphoto.com/video/yoga-fitness-and-woman-at-a-beach-with-warrior-praying-hands-or-stretching-for-gm1856343710-552259858?utm_source=pixabay&utm_medium=affiliate&utm_campaign=SRP_video_sponsored_ratiochange&utm_content=https%3A%2F%2Fpixabay.com%2Fvideos%2Fsearch%2Fyoga%2F&utm_term=yoga"

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process the video frame by frame
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe Pose
                results = pose.process(frame_rgb)

                # Draw pose landmarks on the frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                    )

                # Write the processed frame to the output video
                out.write(frame)

                # Display the frame (optional)
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processed video saved to {output_path}")

        result = {"status": "ok cv", "data":{"video":output_video}}

        return result

