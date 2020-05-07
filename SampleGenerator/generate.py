import pathlib
from typing import List, Tuple, Dict 
import os
import numpy as np
from PIL import Image

from detector import detect_faces

def generate_dataset(root_folder_path):
	original_images = list(pathlib.Path(root_folder_path).glob('*'))

	for i, img_path in enumerate(original_images):
		image = np.array(Image.open(img_path).convert("RGB"))
		detect_faces(image)


def main():
	generate_dataset(root_folder_path="test_images")

if __name__ == "__main__":
	main()