import random

import mediapipe as mp
from .mediapipeIndices import mediapipeIndices
from .eyesClassificatorWrapper import EyesWrapper
import numpy as np
import json
import math
import cv2

config = json.load(open("config.json"))

class FaceMeshWrapper:
    def __init__(self, eyeThresh = 5.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.eyeThresh = eyeThresh
        self.eyeWrapper = EyesWrapper(
            config["eyes_model_weight_path"],
            config["eye_model_thresh"]
        )

    def drawLandmarks(self, image, results):
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        return annotated_image

    def predictFaceMesh(self, image : np.ndarray):

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2
        ) as face_mesh:
            results = face_mesh.process(image)

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                return None
            else:
                return results

    def landmarksDetection(self, image, results):
        img_height, img_width = image.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]

        # returning the list of tuples for each landmark
        return mesh_coord

    # Euclaidean distance
    def euclaideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    def separate_eye(self, image, mesh, eye_indices):
        if mesh is None:
            return mesh
        landmarks = self.landmarksDetection(image, mesh)
        x_coords = []
        y_coords = []

        for index in eye_indices:
            x, y = list(landmarks[index])
            x_coords.append(x)
            y_coords.append(y)

        min_y, max_y = min(y_coords), max(y_coords)
        min_x, max_x = min(x_coords), max(x_coords)

        delta_y, delta_x = max_y - min_y, max_x - min_x
        delta = int(max(delta_x, delta_y) * 0.75)

        y_mean, x_mean = (min_y+max_y)/2, (min_x+max_x)/2
        max_y = min(image.shape[0], int(y_mean + delta))
        min_y = max(0, int(y_mean - delta))
        max_x = min(image.shape[1], int(x_mean + delta_x))
        min_x = max(0, int(x_mean - delta))
        eye = image[min_y:max_y, min_x:max_x]
        return eye

    # Blinking Ratio
    def blinkRatio(self, image, landmarks, right_indices, left_indices):
        """
        :
        This function, selects the landmarks for horizontal points,
        and vertical points of Eyes, and find the distance between point,
        with help of Euclidean Distance, and calculates blink ratio for each,
        by dividing the horizontal distance with vertical disntance,
        which allow us to detect Blink Of Eye.

        :param image: input image
        :param landmarks: are mesh_coords which return by landmarks detector funtion
        :param right_indices: these are nothing but landmarks on Face Mesh, Right Eyes
        :param left_indices: these are nothing but landmarks on Face Mesh, Left Eyes

        :return: combined ratio of both Eyes, which allows us to detect blinks
        """
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes
        cv2.line(image, rh_right, rh_left, (0, 255, 0), 2)
        cv2.line(image, rv_top, rv_bottom, (255, 255, 255), 2)
        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]
        # Finding Distance Right Eye
        rhDistance = self.euclaideanDistance(rh_right, rh_left)
        rvDistance = self.euclaideanDistance(rv_top, rv_bottom)
        # Finding Distance Left Eye
        lvDistance = self.euclaideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclaideanDistance(lh_right, lh_left)
        # Finding ratio of LEFT and Right Eyes
        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance

        return image, leRatio, reRatio

    def eyesOpened(self, image, mesh) -> (bool, bool):
        if mesh:
            landmarks = self.landmarksDetection(image, mesh)
            image, leRatio, reRatio = self.blinkRatio(
                image,
                landmarks,
                mediapipeIndices.RIGHT_EYE,
                mediapipeIndices.LEFT_EYE,
            )
            return leRatio <= self.eyeThresh, reRatio <= self.eyeThresh
        return False, False

    def __call__(self, image):
        annotations = dict()
        mesh = self.predictFaceMesh(image)

        left_eye_opened, right_eye_opened = self.eyesOpened(image, mesh)
        # left_eye = self.separate_eye(image, mesh, mediapipeIndices.LEFT_EYE)
        # right_eye = self.separate_eye(image, mesh, mediapipeIndices.RIGHT_EYE)

        # left_eye_opened = self.eyeWrapper(left_eye)
        # right_eye_opened = self.eyeWrapper(right_eye)
        annotations['leftEyeOpened'] = left_eye_opened
        annotations['rightEyeOpened'] = right_eye_opened

        return annotations