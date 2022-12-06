import cv2
import numpy as np
import mediapipe as mp


class MediapipeHandEstimator:
    """
    _summary_ : This class predicts the 21 3D joint locations
    """

    def __init__(self, enable_vis=True):
        # 1. Load model
        self.enable_vis = enable_vis
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def predict_3d_joints(self, frame):
        with self.mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
        ) as hands:
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detections
            results = hands.process(image)
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Detections
            joints_3d = np.array([])
            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in self.mp_hands.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        joints_3d = np.append(
                            joints_3d,
                            [
                                normalizedLandmark.x,
                                normalizedLandmark.y,
                                normalizedLandmark.z,
                            ],
                        )

            # Rendering results
            if self.enable_vis:
                if results.multi_hand_landmarks:
                    for _, hand in enumerate(results.multi_hand_landmarks):
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(
                                color=(192, 192, 192), thickness=2, circle_radius=2
                            ),
                            self.mp_drawing.DrawingSpec(
                                color=(0, 0, 0), thickness=2, circle_radius=2
                            ),
                        )
                        # Drawing 3D
                        # self.mp_drawing.plot_landmarks(hand, self.mp_hands.HAND_CONNECTIONS, azimuth=5)
                    cv2.imshow("Hand pose estimate", image)
                    cv2.waitKey(15)
            return joints_3d


if __name__ == "__main__":
    pose_estimator = MediapipeHandEstimator()

    # Get image from web camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        joints_3d = pose_estimator.predict_3d_joints(frame)
        print(joints_3d[0:3])
