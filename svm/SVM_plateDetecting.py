import numpy as np
import cv2

# 전처리 함수
def preprocessing(car_no):
    image = cv2.imread("images/car/%02d.jpg" % car_no, cv2.IMREAD_COLOR)            # 이미지 파일 불러오기
    if image is None:
        return None, None
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 이미지를 그레이 스케일로 변환
    return gray_img, image

# 슬라이딩 윈도우 탐지 함수
def sliding_window_detection(gray_img, win_size, step_size, svm, y_start_ratio):
    detections = []
    y_start = int(gray_img.shape[0] * y_start_ratio)        # 영상 아래 부분만 검색

    # 높이와 너비를 순회하여 창 이동시키기
    for y in range(y_start, gray_img.shape[0] - win_size[1], step_size):
        for x in range(0, gray_img.shape[1] - win_size[0], step_size):
            # 현재 창의 이미지 추출
            win = gray_img[y:y + win_size[1], x:x + win_size[0]]
            # 추출한 이미지를 (144, 28) 크기로 리사이즈
            resized_win = cv2.resize(win, (144, 28), interpolation=cv2.INTER_AREA)
            # 리사이즈한 이미지를 1차원 배열로 변환하고 데이터 타입을 np.float32로 설정
            resized_win = resized_win.reshape(-1).astype(np.float32)
            # SVM을 사용해 현재 창의 이미지 분류
            _, pred = svm.predict(resized_win.reshape(1, -1))

            # 예측 결과가 1인 경우만 검출 목록에 추가합니다.
            if pred == 1:
                detections.append((x, y, win_size[0], win_size[1]))
    return detections


def main():
    svm = cv2.ml.SVM_create()               # SVM 객체를 생성하고 학습된 모델 불러오기
    svm = svm.load("SVMtrain.xml")
    if svm.empty():
        print("Failed to load SVM model.")
        return
    img_idx = [0, 1, 2, 3, 5, 9, 12, 13, 14]    # 임의의 자동차 파일 선택

    # 다양한 사이즈의 슬라이딩 창 지정
    sliding_sizes = [
        [160, 30], [160, 40], [160, 50],
        [170, 30], [170, 40], [170, 50],
        [180, 30], [180, 40], [180, 50],
        [190, 30], [190, 40], [190, 50],
        [200, 30], [200, 40], [200, 50],
        [210, 40], [210, 50], [210, 50],
        [220, 40], [220, 50], [220, 50]
    ]

    # 값이 작을수록 높이와 너비를 더 세밀하게 탐지함, 대신 느려짐
    step_size = 4

    # 영상의 반틈 아래 부분만 검색
    y_start_ratio = 0.5

    for car_no in img_idx:
        gray_img, image = preprocessing(car_no)
        if image is None:
            continue

        # 각 슬라이딩 창 크기에 대해 이미지에서 객체 검출
        for win_size in sliding_sizes:
            rectangles = sliding_window_detection(gray_img, win_size, step_size, svm, y_start_ratio)

            # 검출된 객체들을 화면에 표시
            for x, y, w, h in rectangles:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 검출된 객체가 표시된 이미지를 화면에 출력
        cv2.imshow("Detected Objects", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
