import json
import cv2
import argparse
from pprintjson import pprintjson
import sys
import os

from wrappers import PipeWrapper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_image', nargs='+', type=str, help='path to the image')
    parser.add_argument('--path_to_config', nargs='+', type=str, default='config.json', help='path to config.json')
    opt = parser.parse_args()

    config = json.load(open(opt.path_to_config))

    # initialize wrapper
    pipe = PipeWrapper(config)

    # go through all filenames we have
    for filename in opt.path_to_image:
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
