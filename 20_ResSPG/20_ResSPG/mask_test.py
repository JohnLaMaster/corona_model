import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from config import opt
from utils.anchor import get_anchor, get_bounding_box, get_max, detect_red

# def detect_red(img_dir):
#     lower2 = np.array([78, 43, 46])
#     upper2 = np.array([124, 255, 255])
#     img = cv2.imread(img_dir)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     mask = cv2.inRange(hsv, lower2, upper2)
#     img[mask == 0] = [0, 0, 0]
#     return img


def rectangle_mask(img_dir):
    img = cv2.imread(img_dir)
    threshold_img = img[:, :, 0]

    # img_max = np.max(threshold_img)
    #
    # loc = np.where(img_max == threshold_img)
    # print(len(loc[0]))

    # threshold_img = np.stack([threshold_img] * 3, 2)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # label_img = np.zeros(img.shape, dtype=np.int32)
    # cv2.drawContours(label_img, contours, -1, (255, 255, 0), thickness=cv2.FILLED)

    ret = threshold_img < 1.3 * np.mean(threshold_img)
    # threshold = 1.3 * np.mean(img)
    img[ret] = [0, 0, 0]
    #img[threshold_img> ret] = [255, 255, 0]
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def get_box(img, x, y, anchor):
    anchor[:, 0] += y
    anchor[:, 2] += y
    anchor[:, 1] += x
    anchor[:, 3] += x
    anchor = anchor.astype(np.int32)
    for anc in anchor:
        y0, x0, y1, x1 = anc
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)
    return img

img_path = r'D:\bypy\05207a0f-a1bc-46f8-8acb-aa72197e52af.jpg'
x = int(1024 * 0.28)
y = int(1024 * 0.4)

img = cv2.imread(img_path)
# plt.figure(figsize=(12, 12))
# plt.imshow(img)
# plt.show()
img1 = detect_red(img)
# cv2.circle(img1, (x, y), 10, (0, 255, 0), -1)
# plt.figure(figsize=(12, 12))
# plt.imshow(img1)
# plt.show()
# img2 = get_box(img1.copy(), x, y, opt.anchor)
# plt.figure(figsize=(12, 12))
# plt.imshow(img2)
# plt.show()

box = get_bounding_box(img, x, y, opt.anchor.copy(), opt.stride)
y0, x0, y1, x1 = box
cv2.rectangle(img1, (x0, y0), (x1, y1), (0, 0, 255), 5)
plt.figure(figsize=(12, 12))
plt.imshow(img1)
plt.show()

