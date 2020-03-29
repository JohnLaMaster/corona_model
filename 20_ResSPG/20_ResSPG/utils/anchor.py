import numpy as np
import math
import cv2


def get_anchor(ratios, scales):
    anchor = []
    for j, scal in enumerate(scales):
        for i, rat in enumerate(ratios):
            half_w = scal * rat / (math.sqrt(rat) * 2)
            half_h = scal * math.sqrt(rat) / 2
            y0 = 0 - half_h
            x0 = 0 - half_w
            y1 = half_h
            x1 = half_w
            anchor.append([y0, x0, y1, x1])
            anchor.append([x0, y0, x1, y1])

            half_w = scal * rat / 2
            half_h = scal / 2
            y0 = 0 - half_h
            x0 = 0 - half_w
            y1 = half_h
            x1 = half_w
            anchor.append([y0, x0, y1, x1])
            anchor.append([x0, y0, x1, y1])

        half_w = scal / 2
        half_h = scal / 2
        y0 = 0 - half_h
        x0 = 0 - half_w
        y1 = half_h
        x1 = half_w
        anchor.append([y0, x0, y1, x1])
    return np.stack(anchor)


def detect_red(img):
    lower2 = np.array([50, 43, 46])
    upper2 = np.array([124, 255, 255])
    img1 = img.copy()
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower2, upper2)
    img1[mask != 0] = [0, 0, 0]
    img1[mask == 0] = [255, 0, 0]
    return img1


def get_bounding_box(img, x, y, anchor, stride):
    anchor[:, 0] += y
    anchor[:, 2] += y
    anchor[:, 1] += x
    anchor[:, 3] += x
    anchors = anchor.astype(np.int32)
    idx = get_max(img, anchors)
    return anchors[idx, :]


def get_max(img, anchor):
    img1 = detect_red(img)
    score = np.zeros(anchor.shape[0])
    for i, ah in enumerate(anchor):
        y0, x0, y1, x1 = ah
        score[i] = np.mean(img1[y0:y1, x0:x1, 0]) * math.pow((y1 - y0) * (x1 - x0) / 100, 0.25)
    score = np.nan_to_num(score)
    idx = np.argmax(score)
    return idx
