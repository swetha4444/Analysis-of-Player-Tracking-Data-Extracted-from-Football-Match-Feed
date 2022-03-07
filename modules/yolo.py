import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {0: 'player', 1: 'ball'}


class YOLO():
    def __init__(self,model_path, conf_thres, iou_thres):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        print("Yolo model loaded!")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self,frame):
        """
            Input :
                    BGR image
                 
            Output:
            yolo return list of dict in format:
                {   label   :  str
                    bbox    :  [(xmin,ymin),(xmax,ymax)]
                    score   :  float
                    cls     :  int
                }
        """
        img = cv2.resize(frame, (640,384))

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=0)
        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(),2)
                label = classes[int(p[5])]
                xmin = int(p[0] * frame.shape[1] /640)
                ymin = int(p[1] * frame.shape[0] /384)
                xmax = int(p[2] * frame.shape[1] /640)
                ymax = int(p[3] * frame.shape[0] /384)
                item = {'label': label,
                        'bbox' : [(xmin,ymin),(xmax,ymax)],
                        'score': score,
                        'cls' : int(p[5])}
                items.append(item)

        return(items)

# pallete = {'b': (0, 0, 128),
#         'g': (0, 128, 0),
#         'r': (255, 0, 0),
#         'c': (0, 192, 192),
#         'm': (192, 0, 192),
#         'y': (192, 192, 0),
#         'k': (0, 0, 0),
#         'w': (255, 255, 255)}

# color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# def compute_color_for_labels(label):
#     """
#     Simple function that adds fixed color depending on the class
#     """
#     color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
#     return tuple(color)


# def draw_boxes(img, yoloOutput, identities=None, offset=(0, 0)):
#     for det in yoloOutput:
#         box = det["bbox"]
#         x1, y1, x2, y2 = box[0][0],box[0][1],box[1][0],box[1][1]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id =  0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img