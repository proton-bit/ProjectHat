import json
import cv2
import argparse
from pprintjson import pprintjson
import sys
import os

from wrappers import PipeWrapper

def annotate(filename):
    supported_extenstions = [".jpg", ".png", '.jpeg']
    if not any([path.endswith(extension) for extension in supported_extenstions]):
        assert f"Unsupported file format {filename[filename.index('.'):]}"
    print(filename)
    image = cv2.imread(filename)

    path_to_annotation = f"{filename[:filename.index('.')]}.json".replace(
        config['images_directory'],
        config['annotations_directory']
    )

    # obtain annotations from an image
    annotations = pipe(
        image, path_to_annotation
    )
    # display data to console in json format
    pprintjson(annotations)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image', nargs='+', type=str, help='path to the image')
    parser.add_argument('--path_to_config', nargs='+', type=str, default='config.json', help='path to config.json')
    opt = parser.parse_args()

    config = json.load(open(opt.path_to_config))
    # initialize wrapper
    pipe = PipeWrapper(config)

    # go through all filenames we have
    for path in opt.path_to_image:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if not filename.startswith("."):
                    annotate(os.path.join(path, filename))
        else:
            annotate(path)

