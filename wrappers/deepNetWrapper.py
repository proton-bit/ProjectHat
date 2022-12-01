import os

import numpy as np
from deepface import DeepFace
import tempfile
import cv2
import sys

def annotate_face(image: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as ftmp:
        cv2.imwrite(ftmp.name, image)

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        face_annotations = DeepFace.analyze(
            img_path=ftmp.name,
            actions=('age', 'gender', 'race', 'emotion'),
            enforce_detection=False,
            prog_bar=False
        )

        sys.stdout = old_stdout
        return face_annotations