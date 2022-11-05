import json

from wrappers import PipeWrapper
import argparse
import cv2
import os

def print_annotations(annotation, indent = 0):
    for key, value in annotation.items():
        if isinstance(value, dict):
            print('\t' * indent + (("%s: {") % str(key)))
            print_annotations(value, indent + 1)
            print('\t' * indent + ' ' + ('}'))
        elif isinstance(value, list):
            for val in value:
                print('\t' * indent + (("%s: [\n") % str(key)))
                print_annotations(val, indent + 1)
        else:
            print('\t' * indent + (("%s: %s") % (str(key), str(value))))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config', nargs='+', type=str, default='config.json', help='path to config.json')
    opt = parser.parse_args()

    config = json.load(open(opt.path_to_config))

    pipe = PipeWrapper(config)
    for filename in os.listdir(config["images_directory"]):
        # to prevent by reading files such as .DS_Store etc...
        if filename.startswith("."):
            continue

        path_to_image = os.path.join(
            config["images_directory"],
            filename
        )
        image = cv2.imread(path_to_image)

        path_to_annotation = f"{path_to_image[:path_to_image.index('.')]}.json".replace(
            config['images_directory'],
            config['annotations_directory']
        )

        annotations = pipe(
            image, path_to_annotation
        )

        print(f"\n{'-'*40}\n")
        print(f"{path_to_image}")
        print_annotations(annotations)
        print(f"\n{'-' * 40}\n")