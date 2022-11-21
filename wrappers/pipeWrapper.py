from .faceMeshWrapper import FaceMeshWrapper
from .headWrapper import HeadDetector
from .blurDetection import BlurDetector
import json
import cv2


class PipeWrapper:
    def __init__(self, config):

        self.max_aspect_ratio = config['max_aspect_ratio']
        self.detector_conf_thresh = config['detector_conf_thresh']
        self.detector = HeadDetector(
            config["detector_weights_path"],
            config["yaml_config_path"],
        )
        self.blurDetector = BlurDetector(config["blur_thresh"])
        self.faceMesh = FaceMeshWrapper(config["face_mesh_eye_thresh"])

        self.findArea = lambda a: abs(a[2] - a[0]) * abs(a[3] - a[1])

    def __call__(self, image, path_to_annotations="annotations/annotation.json"):
        print()
        # inference head-detector
        coords, _ = self.detector(image, conf_thres=self.detector_conf_thresh)
        # sort coords by bbox area in descending order
        coords.sort(key=self.findArea, reverse=True)
        # remove samples that too small relatively to the neighbour
        # neighbour in this case is the closest by area of the box
        for index in range(len(coords) - 1):
            first_area, second_area = self.findArea(coords[index]), self.findArea(coords[index + 1])
            # if aspect ratio too big we slice coords
            if first_area / second_area > self.max_aspect_ratio:
                coords = coords[:index + 1]
                break

        # Data to be written
        dictionary = dict(
            number_of_persons=len(coords),
            blurry=bool(),
            persons=dict()
        )
        # cv2.imshow("image", image)
        dictionary["blurry"] = self.blurDetector(image)

        for index, coord in enumerate(coords):
            # expand coordinates
            # expand_percentage = 0.1
            # coord[0] = max(coord[0] - abs(coord[2] - coord[0]) * expand_percentage, 0)
            # coord[1] = max(coord[1] - abs(coord[3] - coord[1]) * expand_percentage, 0)
            # coord[2] = min(coord[2] + abs(coord[2] - coord[0]) * expand_percentage, image.shape[0])
            # coord[3] = min(coord[3] + abs(coord[3] - coord[1]) * expand_percentage, image.shape[1])
            # coord = list(map(int, coord))
            # print(coord)

            coord = [
                max(0, coord[0]),
                max(0, coord[1]),
                max(image.shape[1], coord[2]),
                max(image.shape[0], coord[3]),
            ]

            # crop each specific head from the image
            head = image[coord[1]:coord[3], coord[0]:coord[2]]

            # estimate face landmarks and annotations
            head_annotations = self.faceMesh(head)
            # estimate if the specific head is blurry
            blurry = self.blurDetector(head)
            # add information about blur of the head to annotations
            head_annotations['blurry'] = blurry
            # write annotation to json file
            dictionary["persons"][f"person_{index + 1}"] = head_annotations
            # display face if eyes closed
            if not (head_annotations["leftEyeOpened"] and head_annotations["rightEyeOpened"]):
                drawn_head = head
                results = self.faceMesh.predictFaceMesh(drawn_head)
                try:
                    drawn_head = self.faceMesh.drawLandmarks(drawn_head, results)
                except AttributeError:
                    print("no landmarks")
                cv2.imshow(f"head_{index} left={head_annotations['leftEyeOpened']}, right={head_annotations['rightEyeOpened']}", drawn_head)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Serializing json
        json_object = json.dumps(dictionary, indent=4)

        # Writing to sample.json
        with open(path_to_annotations, "w") as outfile:
            outfile.write(json_object)

        return dictionary
