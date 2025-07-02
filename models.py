

import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder
from ultralytics import YOLOWorld
from utils.config import YOLO_WORLD_PATH
import cv2

class YoloWorld():
    
    def __init__(self):
        self.model = YOLOWorld(YOLO_WORLD_PATH)
        self.classes = None

    def set_classes(self, classes):
        self.model.set_classes(classes)

    def predict(self, image):
        results = self.model.predict(image, verbose=False, conf=0.1)
        bboxes = []
        classes = []
        confidences = []
        for result in results:
            detections = result.boxes.cpu().numpy()  
            for detection in detections:                
                bbox = detection.xyxy[0]
                # Convert the class index to integer before using it as an index
                class_idx = int(detection.cls[0])
                id_class = result.names[class_idx]
                confidence = detection.conf[0]
                bboxes.append(bbox)
                classes.append(id_class)
                confidences.append(confidence)
        return bboxes, classes, confidences

    def get_image_with_bboxes(self, image, conf=0.1):
        bboxes, classes, confidences = self.predict(image)
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            class_id = classes[i]
            confidence = confidences[i]
            if confidence > conf:
                cv2.putText(image, class_id + " "+ str(confidence), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,  1, (250,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (250,0,0), 2)
        return image


class VitSam():

    def __init__(self, encoder_model, decoder_model):
        self.decoder = SamDecoder(decoder_model)
        self.encoder = SamEncoder(encoder_model)


    def __call__(self, img, bboxes):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = self._preprocess(raw_img, img_size=512)
        img_embeddings = self.encoder(img)
        boxes = np.array(bboxes, dtype=np.float32)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )

        return masks, boxes

    def _preprocess(self, x, img_size=512):
        pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

        x = torch.tensor(x)
        resize_transform = SamResize(img_size)
        x = resize_transform(x).float() / 255
        x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

        h, w = x.shape[-2:]
        th, tw = img_size, img_size
        assert th >= h and tw >= w
        x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

        return x
