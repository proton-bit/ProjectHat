from wrappers import BlurDetector
import albumentations as A
import cv2
import os
import json

motionBlur = A.MotionBlur(blur_limit=(50, 51), p=1.0)
blurDetector = BlurDetector(60)

config = json.load(open('config.json'))
for filename in os.listdir(config["images_directory"]):
    if filename.startswith("."):
        continue
    filepath = os.path.join(config["images_directory"], filename)
    image = cv2.imread(filepath)

    # gauss_image = A.gaussian_blur(image, 21)
    motioned_image = motionBlur(image=image)["image"]
    blur_map, score, is_blurry = blurDetector(motioned_image)
    print(
        blurDetector(image)[1:],
        f"{(score, is_blurry)}"
    )

    cv2.imshow(filename, image)
    cv2.imshow("blurry", motioned_image)
    cv2.imshow("blurMap", blur_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
