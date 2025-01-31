"""
This script generates COCO JSON file for custom datasets.

For the COCO format, please refer to the official documentation: https://cocodataset.org/#format-data

For image "file_name", use full path to the image file, e.g., "data/iwp/train/0001.jpg".
"""

import argparse
import json
from datetime import date

import cv2


def preprocess_iwp_data():
    for split in ["train", "val", "test"]:
        data = json.load(open("data/iwp/{}/via_region_data.json".format(split)))

        # crate coco format
        coco = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # info
        coco["info"] = {
            "description": "IWP dataset",
            "date_created": "{}".format(date.today()),
        }

        # categories
        coco["categories"] = [{"id": 1, "name": "iwp", "supercategory": "iwp"}]

        # images and annotations
        image_id = 0
        annotation_id = 0
        for _, v in data.items():
            image_id += 1
            # load image to check width and height
            img = cv2.imread("data/iwp/{}/{}".format(split, v["filename"]))
            height, width, _ = img.shape

            image = {
                "id": image_id,
                "file_name": "data/iwp/{}/{}".format(split, v["filename"]),
                "width": width,
                "height": height,
            }
            coco["images"].append(image)

            for _, region in v["regions"].items():
                # segmentation
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]
                # transfer to [x1, y1, x2, y2, ...]
                segmentation = [
                    item
                    for sublist in zip(all_points_x, all_points_y)
                    for item in sublist
                ]
                # check segmentation for points less than 6
                if len(segmentation) < 6:
                    print(
                        "Warning: Image {} has a mask less than 3 points. This is not allowed in COCO format. This mask will be skipped.".format(
                            v["filename"]
                        )
                    )
                    continue

                annotation_id += 1

                # find bounding box, add 1 as buffer
                x = min(all_points_x) - 1
                y = min(all_points_y) - 1
                w = max(all_points_x) + 1 - x
                h = max(all_points_y) + 1 - y

                # area
                area = w * h

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
                coco["annotations"].append(annotation)

        # save to file
        with open("data/iwp/iwp_{}.json".format(split), "w") as f:
            json.dump(coco, f)


def preprocess_iwp_multi():
    for split in ["train", "val", "test"]:
        data = json.load(open("data/iwp/{}/via_region_data.json".format(split)))

        # crate coco format
        coco = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # info
        coco["info"] = {
            "description": "Multi-class IWP dataset",
            "date_created": "{}".format(date.today()),
        }

        # categories
        coco["categories"] = [
            {"id": 1, "name": "lowcenter", "supercategory": "iwp"},
            {"id": 2, "name": "highcenter", "supercategory": "iwp"},
        ]

        # images and annotations
        image_id = 0
        annotation_id = 0
        for _, v in data.items():
            image_id += 1
            # load image to check width and height
            img = cv2.imread("data/iwp_multi/{}/{}".format(split, v["filename"]))
            height, width, _ = img.shape

            image = {
                "id": image_id,
                "file_name": "data/iwp_multi/{}/{}".format(split, v["filename"]),
                "width": width,
                "height": height,
            }
            coco["images"].append(image)

            for _, region in v["regions"].items():
                # segmentation
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]
                # transfer to [x1, y1, x2, y2, ...]
                segmentation = [
                    item
                    for sublist in zip(all_points_x, all_points_y)
                    for item in sublist
                ]
                # check segmentation for points less than 6
                if len(segmentation) < 6:
                    print(
                        "Warning: Image {} has a mask less than 3 points. This is not allowed in COCO format. This mask will be skipped.".format(
                            v["filename"]
                        )
                    )
                    continue

                annotation_id += 1

                # find bounding box, add 1 as buffer
                x = min(all_points_x) - 1
                y = min(all_points_y) - 1
                w = max(all_points_x) + 1 - x
                h = max(all_points_y) + 1 - y

                # area
                area = w * h

                # category
                category = region["region_attributes"]["object_name"]
                if category == "lowcenter":
                    category_id = 1
                elif category == "highcenter":
                    category_id = 2
                else:
                    raise ValueError("Unknown category: {}".format(category))

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
                coco["annotations"].append(annotation)

        # save to file
        with open("data/iwp_multi/iwp_multi_{}.json".format(split), "w") as f:
            json.dump(coco, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate COCO JSON file for custom datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    args = parser.parse_args()

    if args.dataset == "iwp":
        preprocess_iwp_data()
    elif args.dataset == "iwp_multi":
        preprocess_iwp_multi()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
