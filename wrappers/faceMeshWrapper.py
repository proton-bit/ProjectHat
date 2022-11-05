import mediapipe as mp
from .mediapipeIndices import mediapipeIndices
import numpy as np
import math
import cv2

class FaceMeshWrapper:
    def __init__(self, eyeThresh = 5.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.eyeThresh = eyeThresh

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
            min_detection_confidence=0.5
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
        # cv.line(image, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(image, rv_top, rv_bottom, utils.WHITE, 2)
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
        return leRatio, reRatio

    def eyesOpened(self, image, mesh) -> bool:
        if mesh:
            landmarks = self.landmarksDetection(image, mesh)
            ratios = self.blinkRatio(
                image,
                landmarks,
                mediapipeIndices.RIGHT_EYE,
                mediapipeIndices.LEFT_EYE,
            )

            if sum(ratios) / 2 <= self.eyeThresh:
                return True

        return False

    def __call__(self, image):
        annotations = dict()

        if image.shape != (256, 256, 3):
            image = cv2.resize(image, (256, 256))
        mesh = self.predictFaceMesh(image)
        annotations['eyeOpened'] = self.eyesOpened(image, mesh)

        return annotations