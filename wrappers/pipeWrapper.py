from .faceMeshEstimatorWrapper import FaceMeshWrapper
from .headDetectorWrapper import HeadDetector
from .blurDetection import BlurDetector
from .deepNetWrapper import annotate_face
import json


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
        """
        Computes annotations like:

        :param image: input image
        :param path_to_annotations: path to save json of type
        :return: string with json like:
        {
            "number_of_persons": 2,
            "blurry": true,
            "persons": [
                {
                    "leftEyeOpened": true,
                    "rightEyeOpened": true,
                    "age": 31,
                    "region": {
                        "x": 0,
                        "y": 0,
                        "w": 173,
                        "h": 152
                    },
                    "gender": "Man",
                    "race": {
                        "asian": 2.903786115348339,
                        "indian": 6.636875867843628,
                        "black": 5.828463658690453,
                        "white": 42.43169128894806,
                        "middle eastern": 23.266273736953735,
                        "latino hispanic": 18.932905793190002
                    },
                    "dominant_race": "white",
                    "emotion": {
                        "angry": 4.450730555589574e-12,
                        "disgust": 2.840043101688364e-27,
                        "fear": 4.433291002258976e-22,
                        "happy": 100.0,
                        "sad": 1.1921515241186762e-12,
                        "surprise": 5.9187781859206005e-19,
                        "neutral": 4.1958706731293205e-06
                    },
                    "dominant_emotion": "happy",
                    "blurry": true
                },
                {
                    "leftEyeOpened": true,
                    "rightEyeOpened": true,
                    "age": 29,
                    "region": {
                        "x": 0,
                        "y": 0,
                        "w": 155,
                        "h": 135
                    },
                    "gender": "Man",
                    "race": {
                        "asian": 14.501644670963287,
                        "indian": 17.828726768493652,
                        "black": 15.888991951942444,
                        "white": 12.507160007953644,
                        "middle eastern": 13.513036072254181,
                        "latino hispanic": 25.76044201850891
                    },
                    "dominant_race": "latino hispanic",
                    "emotion": {
                        "angry": 2.8611292017145287e-11,
                        "disgust": 1.4329748206212888e-18,
                        "fear": 3.508866866779898e-16,
                        "happy": 99.99995827674617,
                        "sad": 5.447459214917005e-10,
                        "surprise": 4.646206385586971e-12,
                        "neutral": 4.417319111774691e-05
                    },
                    "dominant_emotion": "happy",
                    "blurry": true
                }
            ]
        }

        """
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
            persons=[]
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

            # crop each specific head from the image
            head = image[coord[1]:coord[3], coord[0]:coord[2]]
            # estimate face landmarks and annotations
            head_annotations = self.faceMesh(head)
            # obtain features and expressions from face
            head_annotations.update(annotate_face(head))
            # estimate if the specific head is blurry
            blurry = self.blurDetector(head)
            # add information about blur of the head to annotations
            head_annotations['blurry'] = blurry
            # write annotation to json file
            dictionary["persons"].append(head_annotations)
            # display face if eyes closed

            # if not (head_annotations["leftEyeOpened"] and head_annotations["rightEyeOpened"]):
            #     drawn_head = head
            #     results = self.faceMesh.predictFaceMesh(drawn_head)
            #     try:
            #         drawn_head = self.faceMesh.drawLandmarks(drawn_head, results)
            #     except AttributeError:
            #         print("no landmarks")
            #     cv2.imshow(f"head_{index}", drawn_head)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        # Serializing json
        json_object = json.dumps(dictionary, indent=4)

        # Writing to sample.json
        with open(path_to_annotations, "w") as outfile:
            outfile.write(json_object)

        return dictionary
