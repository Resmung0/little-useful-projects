# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "imagehash>=4.3.1",
#     "numpy>=1.21.0",
#     "opencv-contrib-python",
#     "pillow>=9.0.0",
#     "termcolor>=2.2.0",
#     "tqdm>=4.65.0",
# ]
# ///

#!/usr/bin/env python3

import os
from typing import Any, Literal

import cv2
import numpy as np
from tqdm import tqdm


class ImageSimilarityDetector:
    def __init__(self, threshold=0.7) -> None:
        """
        Initialize the similarity detector

        :param threshold: Similarity threshold (0-1)
        :param use_gpu: Whether to use GPU acceleration
        """
        self.threshold = threshold
        self.orb = cv2.ORB_create()

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Load and preprocess image for comparison

        :param img_path: Path to image file
        :return: Preprocessed image
        """
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Resize to a standard size
        img = cv2.resize(img, (300, 300))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray

    def find_similar_images(self, directory: str) -> list[list[str]]:
        """
        Find similar images in a directory

        :param directory: Path to directory containing images
        :return: list of lists of similar image paths
        """
        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

        # Collect image paths
        image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if os.path.splitext(file.lower())[1] in image_extensions
        ]

        # Prepare descriptors
        print("Computing image descriptors...")
        descriptors = {}
        keypoints = {}

        for path in tqdm(image_paths):
            img = self._preprocess_image(path)
            if img is not None:
                kp, desc = self.orb.detectAndCompute(img, None)
                if desc is not None:
                    descriptors[path] = desc
                    keypoints[path] = kp

        # Find similar images
        similar_groups = []
        checked = set()

        print("Comparing images...")
        for path1 in tqdm(list(descriptors.keys())):
            if path1 in checked:
                continue

            group = [path1]
            checked.add(path1)

            for path2 in descriptors:
                if path2 in checked or path2 == path1:
                    continue

                # Compare descriptors
                similarity = self._compare_descriptors(
                    descriptors[path1], descriptors[path2]
                )

                if similarity > self.threshold:
                    group.append(path2)
                    checked.add(path2)

            if len(group) > 1:
                similar_groups.append(group)

        return similar_groups

    def _compare_descriptors(self, desc1, desc2) -> Any | Literal[0]:
        """
        Compare image descriptors

        :param desc1: First image descriptor
        :param desc2: Second image descriptor
        :return: Similarity score
        """
        # Use FLANN matcher for efficient matching
        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(desc1, desc2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity based on good matches
        # Lower distance means more similar
        if len(matches) > 0:
            # Compute average distance of top matches
            similarity = 1 / (1 + np.mean([m.distance for m in matches[:10]]))
            return similarity

        return 0

    def delete_similar_images(
        self, similar_groups: list[list[str]], keep_mode="first"
    ) -> int:
        """
        Delete similar images

        :param similar_groups: Groups of similar images
        :param keep_mode: Mode for keeping images ('first', 'last', 'random')
        """
        import random

        deleted_count = 0
        for group in similar_groups:
            if len(group) <= 1:
                continue

            # Determine which image to keep
            if keep_mode == "first":
                keep_index = 0
            elif keep_mode == "last":
                keep_index = -1
            else:
                keep_index = random.randint(0, len(group) - 1)

            # Delete other images
            for i, path in enumerate(group):
                if i != keep_index:
                    try:
                        os.remove(path)
                        deleted_count += 1
                        print(f"Deleted: {path}")
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")

        return deleted_count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Image Similarity Detector")
    parser.add_argument(
        "directory", type=str, help="Directory to search for similar images"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold"
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last", "random"],
        default="first",
        help="Which image to keep when finding similar images",
    )
    args = parser.parse_args()

    # Check for CUDA support
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA GPU detected and available!")
    else:
        print("No CUDA GPU detected. Falling back to CPU processing.")

    # Initialize detector
    detector = ImageSimilarityDetector(threshold=args.threshold)

    # Find similar images
    similar_groups = detector.find_similar_images(args.directory)

    # Print similar image groups
    print("\nSimilar Image Groups:")
    for i, group in enumerate(similar_groups, 1):
        print(f"Group {i}: {len(group)} similar images")
        for img in group[:3]:
            print(f"  - {img}")
        if len(group) > 3:
            print(f"    ... and {len(group)-3} more")

    # Confirm deletion
    confirm = input("\nDelete similar images? (y/N): ").lower()
    if confirm == "y":
        deleted = detector.delete_similar_images(similar_groups, args.keep)
        print(f"\nDeleted {deleted} similar images.")


if __name__ == "__main__":
    main()
