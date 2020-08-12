import cv2
import numpy as np


def CountIOU(RecA, RecB):
    xA = max(RecA[0], RecB[0])
    yA = max(RecA[1], RecB[1])
    xB = min(RecA[2], RecB[2])
    yB = min(RecA[3], RecB[3])
    # 计算交集部分面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # 计算预测值和真实值的面积
    RecA_Area = (RecA[2] - RecA[0] + 1) * (RecA[3] - RecA[1] + 1)
    RecB_Area = (RecB[2] - RecB[0] + 1) * (RecB[3] - RecB[1] + 1)
    # 计算IOU
    iou = interArea / float(RecA_Area + RecB_Area - interArea)

    return iou


img = np.zeros((512, 512, 3), np.uint8)
img.fill(255)

RecA = [50, 50, 300, 300]
RecB = [60, 60, 320, 320]

cv2.rectangle(img, (RecA[0], RecA[1]), (RecA[2], RecA[3]), (0, 255, 0), 5)
cv2.rectangle(img, (RecB[0], RecB[1]), (RecB[2], RecB[3]), (255, 0, 0), 5)

IOU = CountIOU(RecA, RecB)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "IOU = %.2f" % IOU, (130, 190), font, 0.8, (0, 0, 0), 2)

cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()
