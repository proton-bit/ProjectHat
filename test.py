import os
import cv2
import json
from wrappers import FaceMeshWrapper

faceMeshWrapper = FaceMeshWrapper()
config = json.load(open("config.json"))
for index, filename in enumerate(os.listdir(config["test_directory"])):
    filepath = os.path.join(config["test_directory"], filename)

    # to avoid .DS_Store file or etc...
    if filename.startswith("."):
        continue

    image = cv2.imread(filepath)
    image = faceMeshWrapper(image)
    cv2.imshow(f"image {index}", image)
    cv2.waitKey(0)