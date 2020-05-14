import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from torchvision import datasets, models, transforms

from models import model_rcnn

THRESHOLD = 0.5


def process_image(image_path, model, device = None):
	image = Image.open(image_path).resize((224,224)).convert("RGB")
	image_np = np.array(image)
	# image = np.moveaxis(image, 2, 0)

	trs = transforms.Compose([transforms.ToTensor()])
	image = trs(image_np)

	if device is not None:
		image = image.to(device)

	with torch.no_grad():
		output = model([image])[0]
		print(output)


	boxes = output["boxes"].cpu().numpy().tolist()
	scores = output["scores"].cpu().numpy().tolist()
	# label = "Mask" if mask > withoutMask else "No Mask"
	# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

	# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
	# cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	
	color = (0, 0, 255)
	for idx, box in enumerate(boxes):
		if scores[idx] >= THRESHOLD:
			print(box)
			cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

	return image_np


def main():
	model = model_rcnn.create_model(2)
	model.load_state_dict(torch.load("model.pt"))
	model.eval()
	processed_image = process_image("./SampleGenerator/test_images/faces_2.jpg", model)
	processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
	cv2.imshow("Output", processed_image)
	cv2.waitKey(0)
	


if __name__ == "__main__":
	main()