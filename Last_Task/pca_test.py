import numpy as np
import cv2

def preprocessing(train_no):
    fname = f"face_img/train/train{train_no:03d}.jpg"
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    cv2.imshow("original image", image)

    # 사용자가 키 입력을 기다립니다.
    cv2.waitKey(0)

    # 사용자가 키를 누르면 모든 창을 닫습니다.
    cv2.destroyAllWindows()

preprocessing(0)
