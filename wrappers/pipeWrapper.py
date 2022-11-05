from .faceMeshWrapper import FaceMeshWrapper
from .headWrapper import HeadDetector
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
            first_area, second_area = self.findArea(coords[index]), self.findArea(coords[index+1])
            # if aspect ratio too big we slice coords
            if first_area / second_area > self.max_aspect_ratio:
                coords = coords[:index+1]
                break

        # Data to be written
        dictionary = {
            "number_of_persons": len(coords),
            "persons": dict()
        }

        cv2.imshow("image", image)
        for index, coord in enumerate(coords):
            # crop each specific head from the image
            head = image[coord[1]:coord[3], coord[0]:coord[2]]
            # estimate face landmarks and annotations
            head_annotations = self.faceMesh(head)
            # write annotation to json file
            dictionary["persons"][f"person_{index + 1}"] = head_annotations
            # display face if eyes closed
            if not head_annotations["eyeOpened"]:
                cv2.imshow(f"head_{index}", cv2.resize(head, (256, 256)))
                cv2.waitKey(1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Serializing json
        json_object = json.dumps(dictionary, indent=4)

        # Writing to sample.json
        with open(path_to_annotations, "w") as outfile:
            outfile.write(json_object)

        return dictionary
