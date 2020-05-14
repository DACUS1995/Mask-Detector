import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import argparse
import imutils
from imutils.video import VideoStream
import time

from models import model_rcnn

THRESHOLD = 0.5


def process_image(image, model, device = None):
	image_np = np.array(image)

	trs = transforms.Compose([transforms.ToTensor()])
	image = trs(image_np)

	if device is not None:
		image = image.to(device)

	with torch.no_grad():
		output = model([image])[0]
		print(output)


	boxes = output["boxes"].cpu().numpy().tolist()
	scores = output["scores"].cpu().numpy().tolist()
	labels = output["labels"].cpu().numpy().tolist()


	class_to_label = {
		2: "normal face", 
		1: "masked face"
	}
	
	for idx, box in enumerate(boxes):
		if scores[idx] >= THRESHOLD:
			print(box)
			label = class_to_label[labels[idx]]
			color = (0, 255, 0) if label == "masked face" else (0, 0, 255)
			cv2.putText(image_np, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 2)
			cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

	return image_np



def main(args):
	model = model_rcnn.create_model(3)
	model.load_state_dict(torch.load("model.pt"))
	model.eval()

	if args.realtime == False:
		image = Image.open(args.image).resize((224,224)).convert("RGB")
		processed_image = process_image(image, model)
		processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
		cv2.imshow("Output", processed_image)
		cv2.waitKey(0)
	else:
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

		while True:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			processed_image = process_image(frame, model)
			cv2.imshow("Frame", processed_image)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break

		cv2.destroyAllWindows()
		vs.stop()
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--realtime", type=bool, default=False, help="run in realtime mode")
	parser.add_argument("--image", type=str, default="./SampleGenerator/test_images/faces.jpg", help="image to detect")
	args = parser.parse_args()
	main(args)