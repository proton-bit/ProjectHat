import cv2
import numpy as np


def fix_image_size(image: np.array, expected_pixels: float = 2E6):
    ratio = np.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


class BlurDetector:
    def __init__(self, threshold: int = 100):
        self.threshold = threshold

    def estimate_blur(self, image: np.array):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return blur_map, score, bool(score < self.threshold)

    def pretty_blur_map(self, blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):

        abs_image = np.abs(blur_map).astype(np.float32)
        abs_image[abs_image < min_abs] = min_abs

        abs_image = np.log(abs_image)
        cv2.blur(abs_image, (sigma, sigma))
        return cv2.medianBlur(abs_image, sigma)

    def __call__(self, image: np.ndarray):
        """
        :param image: any image
        :return: blurry image or not
        """
        resized_image = fix_image_size(image)
        is_blurry = self.estimate_blur(resized_image)[-1]
        return is_blurry
