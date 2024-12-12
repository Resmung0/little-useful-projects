# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "imagehash>=4.3.1",
#     "numpy>=1.21.0",
#     "pillow>=9.0.0",
#     "termcolor>=2.2.0",
#     "tqdm>=4.65.0",
# ]
# ///

#!/usr/bin/env python3

import argparse
import os
import random

import imagehash
from PIL import Image
from termcolor import colored
from tqdm import tqdm


def find_similar_images(directory, threshold=5) -> list:
    """
    Find similar images in the given directory using perceptual hashing.

    :param directory: Path to the directory containing images
    :param threshold: Hamming distance threshold for considering images similar
    :return: List of lists of similar image paths
    """
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    # Collect image paths
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                image_paths.append(os.path.join(root, file))

    # Compute image hashes with progress bar
    print(colored("\n🔍 Scanning images...", "cyan"))
    hashes = {}
    for path in tqdm(image_paths, desc="Hashing", unit="image"):
        try:
            with Image.open(path) as img:
                # Compute perceptual hash
                phash = imagehash.phash(img)
                hashes[path] = phash
        except Exception as e:
            print(colored(f"Error processing {path}: {e}", "yellow"))

    # Find similar images with progress bar
    print(colored("\n🔬 Detecting similar images...", "cyan"))
    similar_images = []
    checked = set()

    paths_to_check = list(hashes.keys())
    for path1 in tqdm(paths_to_check, desc="Comparing", unit="image"):
        if path1 in checked:
            continue

        group = [path1]
        checked.add(path1)

        for path2, hash2 in hashes.items():
            if path2 in checked:
                continue

            # Compare hash using Hamming distance
            if hashes[path1] - hash2 <= threshold:
                group.append(path2)
                checked.add(path2)

        if len(group) > 1:
            similar_images.append(group)

    return similar_images


def delete_similar_images(similar_images, keep_mode="first") -> int:
    """
    Delete similar images based on the specified keep mode.

    :param similar_images: List of lists of similar image paths
    :param keep_mode: Mode for keeping images ('first', 'last', or 'random')
    :return: Number of images deleted
    """
    total_deleted = 0

    print(colored("\n🗑️  Deleting similar images...", "red"))

    # Progress bar for deletion
    for image_group in tqdm(similar_images, desc="Deleting Groups", unit="group"):
        if len(image_group) <= 1:
            continue

        # Determine which images to keep based on mode
        if keep_mode == "first":
            keep_index = 0
        elif keep_mode == "last":
            keep_index = -1
        else:  # random
            keep_index = random.randint(0, len(image_group) - 1)

        # Delete other images
        for i, path in enumerate(image_group):
            if i != keep_index:
                try:
                    os.remove(path)
                    total_deleted += 1
                    print(colored(f"Deleted: {path}", "red"))
                except Exception as e:
                    print(colored(f"Error deleting {path}: {e}", "yellow"))

    return total_deleted


def main() -> None:
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description=colored("🖼️  Find and Delete Similar Images", "green"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory", type=str, help="Directory to search for similar images"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for similarity (default: 5)",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last", "random"],
        default="first",
        help="Which image to keep when finding similar images (default: first)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(colored(f"Error: {args.directory} is not a valid directory.", "red"))
        return

    # Find similar images
    print(colored(f"🔎 Searching for similar images in {args.directory}...", "cyan"))
    similar_images = find_similar_images(args.directory, args.threshold)

    # Print summary of similar image groups
    print(colored("\n📊 Analysis Results:", "green"))
    print(colored(f"Found {len(similar_images)} groups of similar images", "green"))
    for i, group in enumerate(similar_images, 1):
        print(colored(f"Group {i}: {len(group)} similar images", "yellow"))
        for img in group[:3]:  # Show first 3 images in each group
            print(f"  - {img}")
        if len(group) > 3:
            print(colored(f"    ... and {len(group)-3} more", "yellow"))

    # Confirm deletion
    confirm = input(
        colored("\n❓ Do you want to delete similar images? (y/N): ", "cyan")
    ).lower()
    if confirm != "y":
        print(colored("Deletion cancelled.", "yellow"))
        return

    # Delete similar images
    deleted_count = delete_similar_images(similar_images, args.keep)

    # Print final summary
    print(colored("\n✅ Operation Complete!", "green"))
    print(colored(f"Deleted {deleted_count} similar images.", "green"))


if __name__ == "__main__":
    main()
