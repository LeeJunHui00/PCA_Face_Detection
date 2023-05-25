import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocessing(train_no):
    fname = f"face_img/train/train{train_no:03d}.jpg"           # 310개의 train_xxx.jpg 라벨링하기
    image = cv2.imread(fname, cv2.IMREAD_COLOR)                 # 컬러 모드로 이미지 파일 읽기
    resized_image = cv2.resize(image, (120, 150))               # 120x150 사이즈로 픽셀 조정
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)# 그레이 스케일로 변환
    float_image = gray_image.astype(np.float32)                 # 데이터 유형을 float32로 변환
    normalized_image = float_image / 255                        # 픽셀 값을 나누어 정규화
    return normalized_image


def main():
    print("train start")


if __name__ == "__main__":
    main()
    cv2.waitKey(0)