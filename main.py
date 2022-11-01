import os
import cv2
import json
from wrappers import PoseDetector

config = json.load(open("config.json"))


poseDetector = PoseDetector(config["pose_weights"])
for image_file in os.listdir(config["test_directory"]):

    image_path = os.path.join(
        config["test_directory"],
        image_file
    )

    # to avoid .DS_Store file or etc...
    if image_file.startswith("."):
        continue

    print(image_path)
    image = cv2.imread(image_path)

    pred = poseDetector.predict_keypoints(image)

    annotations_path = os.path.join(
            "annotations",
            image_file.replace('jpg', 'json')
    )
    print(f"writing annotations to: {annotations_path}")
    poseDetector.write_annotations(
        image, pred, annotations_path
        )

    image = poseDetector.plot_skeleton(image, pred)

    cv2.imshow(image_file, image)
    cv2.waitKey(0)
