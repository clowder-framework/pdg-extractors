"""
This script loads COCO JSON file and calculates data statistics.
The statistics will be used to set the configuration parameters for each dataset in `configs/data/dataset_name.py`.
The statistics include:
- pixel mean and std
- bounding box statistics: min, max, mean, std
- max number of objects in an image
"""

import argparse
import json

import cv2
import numpy as np


def calculate_statistics(data):
    # use the training set to calculate statistics
    data = json.load(open("data/{}/{}_train.json".format(data, data)))

    # pixel mean and std
    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    for image in data["images"]:
        img = cv2.imread(image["file_name"])
        # covert the data type to float to avoid overflow
        img = img.astype(np.float64)
        sums += np.sum(img, axis=(0, 1))
        squared_sums += np.sum(np.square(img), axis=(0, 1))
        num_pixels += img.shape[0] * img.shape[1]

    print("Pixel mean in BGR order:", sums / num_pixels)
    print(
        "Pixel std in BGR order:",
        np.sqrt(squared_sums / num_pixels - np.square(sums / num_pixels)),
    )

    # bounding box statistics
    min_width, min_height = float("inf"), float("inf")
    max_width, max_height = 0, 0
    sums = np.zeros(2)
    squared_sums = np.zeros(2)
    num_boxes = {}
    max_objects = 0

    for annotation in data["annotations"]:
        if annotation["image_id"] not in num_boxes:
            num_boxes[annotation["image_id"]] = 0
        x, y, w, h = annotation["bbox"]
        min_width = min(min_width, w)
        min_height = min(min_height, h)
        max_width = max(max_width, w)
        max_height = max(max_height, h)
        sums += np.array([w, h])
        squared_sums += np.square([w, h])
        num_boxes[annotation["image_id"]] += 1

    for image_id, num_box in num_boxes.items():
        max_objects = max(max_objects, num_box)

    total_num_boxes = sum(num_boxes.values())

    print(
        "Bounding box width min, max, mean, std:",
        min_width,
        max_width,
        sums[0] / total_num_boxes,
        np.sqrt(
            squared_sums[0] / total_num_boxes - np.square(sums[0] / total_num_boxes)
        ),
    )
    print(
        "Bounding box height min, max, mean, std:",
        min_height,
        max_height,
        sums[1] / total_num_boxes,
        np.sqrt(
            squared_sums[1] / total_num_boxes - np.square(sums[1] / total_num_boxes)
        ),
    )
    print("Max number of objects in an image:", max_objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate data statistics for custom datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    args = parser.parse_args()

    calculate_statistics(args.dataset)
