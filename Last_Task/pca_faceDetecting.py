import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocessing(train_no):
    fname = f"face_img/train/train{train_no:03d}.jpg"           # 310개의 train_xxx.jpg 라벨링하기
    image = cv2.imread(fname, cv2.IMREAD_COLOR)                 # 컬러 모드로 이미지 파일 읽기
    resized_image = cv2.resize(image, (120, 150), cv2.INTER_LINEAR)               # 120x150 사이즈로 픽셀 조정
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)# 그레이 스케일로 변환
    histeq_image = cv2.equalizeHist(gray_image)
    # cv2.imshow("adjusted_image", adjusted_image)
    # cv2.waitKey(0)
    float_image = histeq_image.astype(np.float32)                 # 데이터 유형을 float32로 변환
    normalized_image = float_image / 255                        # 픽셀 값을 나누어 정규화
    return normalized_image

# 이미지 데이터를 전처리한 후 일차원 배열로 변환하여 훈련 데이터 세트를 생성하는 함수
def create_train_data(num_images):
    train_data = np.empty((num_images, 18000))              # 1. 미리 (num_images, 18000) 크기의 빈 배열 생성

    # 2. num_images만큼 반복하면서, 이미지를 전처리한 후 일차원 배열로 변환하여 train_data에 저장
    for train_no in range(num_images):
        preprocessed_image = preprocessing(train_no)        # 2.1 이미지 전처리를 수행
        image_vector = preprocessed_image.flatten()         # 2.2 전처리된 이미지를 18000x1 크기의 1차원 배열로 변환
        train_data[train_no] = image_vector                 # 2.3 train_data에 변환된 1차원 이미지 배열을 저장
    return train_data                                       # 3. 완성된 train_data 반환

# 이미지 배열(faces)의 평균 이미지 벡터를 계산하여 반환
def faces_avg(faces):
    mean_image_vector = np.mean(faces, axis=0)              # 1. 주어진 이미지 배열의 평균을 계산하여 mean_image_vector에 저장
    return mean_image_vector                                # 2. 계산된 평균 이미지 벡터를 반환

# 평균 이미지 벡터를 원래의 이미지 형태(150x120)로 변환한 후, 출력
def faces_avg_show(avg_faces):
    average_img_reshaped = np.reshape(avg_faces, (150, 120))        # 1. 평균 이미지 벡터를 원래의 이미지 형태(150x120)로 재구성
    cv2.imshow('Mean Image', average_img_reshaped)                  # 2. 평균 이미지 출력하기

# 훈련 이미지 데이터와 평균 이미지의 차이를 계산한 후, 차이 이미지를 출력
def show_difference_images(train_data, mean_image):
    num_images = train_data.shape[0]                                # 1. 훈련 데이터에 있는 이미지 개수 계산
    mean_image_reshaped = np.reshape(mean_image, (150, 120))        # 2. 평균 이미지 벡터를 원래의 이미지 형태(150x120)로 재구성

    # 3. 각 이미지와 평균 이미지의 차이를 계산하고 출력
    for i in range(num_images):
        # 3.1 훈련 이미지 벡터를 원래의 이미지 형태로 재구성
        train_image_vector = train_data[i]
        train_image_reshaped = np.reshape(train_image_vector, (150, 120))

        # 3.2 훈련 이미지와 평균 이미지 간의 차이를 계산
        difference_image = np.abs(train_image_reshaped - mean_image_reshaped)

        # 3.3 차이 이미지 출력하기
        cv2.imshow(f'Difference Image {i+1}', difference_image)
        cv2.waitKey(0)

# 훈련 이미지 데이터와 평균 이미지의 차이를 계산한 후, 각 차이 이미지 벡터를 differences 배열에 저장하고 이를 반환
def store_difference_images(train_data, mean_image):
    num_images = train_data.shape[0]                                # 1. 훈련 데이터에 있는 이미지 개수 계산
    differences = np.empty((num_images, 18000))                     # 2. 차이 이미지를 저장할 빈 배열 생성 (num_images, 18000 크기)

    # 3. 각 이미지와 평균 이미지의 차이를 계산하여 differences 배열에 저장
    for i in range(num_images):
        train_image_vector = train_data[i]                           # 3.1 훈련 이미지 벡터를 가져옴
        difference_image_vector = train_image_vector - mean_image    # 3.2 훈련 이미지 벡터와 평균 이미지 벡터의 차이를 계산
        differences[i] = difference_image_vector                     # 3.3 계산된 차이 이미지 벡터를 differences 배열에 저장
    return differences                                               # 4. 완성된 차이 이미지 배열 반환

# 차이 이미지 배열을 전치하고, 그 결과를 반환
def flat_difference_vectors(train_diff):
    flattened_diff = train_diff.T   # 1. 주어진 차이 이미지 배열을 전치(transpose)하여 flattened_diff에 저장
    return flattened_diff           # 2. 전치된 차이 이미지 벡터를 반환

# 전치된 차이 이미지 벡터를 이용하여 공분산 행렬을 계산한 후, 그 결과를 반환
def covariance_matrix(flat_diff):
    cov_matrix = flat_diff @ flat_diff.T    # 1. 주어진 전치된 차이 이미지 벡터를 이용해 공분산 행렬을 계산
    return cov_matrix                       # 2. 계산된 공분산 행렬을 반환

# 공분산 행렬에서 고유값과 고유벡터를 계산한 후, 그 결과를 반환
def eigenvectors_and_eigenvalues(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)      # 1. 주어진 공분산 행렬에서 고유값과 고유벡터를 계산
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    return eigenvalues, eigenvectors                            # 2. 계산된 고유값과 고유벡터를 반환

# 원래 이미지 공간에서의 고유벡터를 계산하기 위해
# 전치된 차이 이미지 배열(train_flat_diff)와 M x M 고유벡터(eigenvectors_MxM)의 행렬곱을 수행한 후, 그 결과를 반환
def compute_eigenvectors_in_original_space(train_flat_diff, eigenvectors_MxM):
    eigenvectors_NxM = train_flat_diff @ eigenvectors_MxM   # 1. 원래 이미지 공간에서의 고유벡터 계산
    return eigenvectors_NxM                                 # 2. 계산된 고유벡터를 반환

#  원래 이미지 공간에서의 고유벡터(eigenvectors_NxM)를 사용하여 고유 얼굴을 생성하고 출력
def display_eigenfaces(eigenvectors_NxM, save=False):
    num_eigenfaces = eigenvectors_NxM.shape[1]                      # 1. 원래 이미지 공간에서의 고유벡터 배열에서 고유 얼굴 개수 계산

    # 2. 각 고유 얼굴 이미지를 출력
    for i in range(num_eigenfaces):
        eigenface_vector = eigenvectors_NxM[:, i]                       # 2.1 고유 얼굴 벡터를 가져옴
        eigenface_image = np.reshape(eigenface_vector, (150, 120))      # 2.2 고유 얼굴 벡터를 원래 이미지 형태(150x120)로 재구성

        # 2.3 명암값을 강조하여 출력하기 쉽게 만듦
        eigenface_image = cv2.normalize(eigenface_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 2.4 고유 얼굴 이미지 출력
        cv2.imshow(f'Eigenface {i + 1}', eigenface_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# 정렬된 고유값(sorted_eigenvalues)을 기준으로 그래프를 그리고 출력
def graph_plot(sorted_eigenvalues):
    plt.figure(figsize=(10, 5))     # 1. 그래프를 그리기 위해 그림의 크기를 설정
    plt.plot(sorted_eigenvalues)    # 2. 정렬된 고유값을 기준으로 그래프 그리기

    # 3. 그래프의 제목, x축 및 y축 라벨 설정
    plt.title('Sorted Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')

    plt.xlim(0, 400)    # 4. x축의 범위 설정
    plt.grid(True)      # 5. 그리드 표시
    plt.show()          # 6. 그래프 출력

# 정렬된 고유값과 원래 이미지 공간에서의 고유벡터을 사용하여 주성분을 찾고
# 목표 설명 분산(K)에 맞게 고유값과 고유벡터를 선택
def find_principal_components(sorted_eigenvalues, eigenvectors_NxM, K=0.88):
    # 1. 고유값의 총 합 구하기
    total_eigenvalues_sum = np.sum(sorted_eigenvalues)
    current_sum = 0
    num_principal_components = 0

    # 2. 주성분의 개수를 결정하여 고유값의 합이 전체 고유값의 합의 K(기본값: 0.95)를 넘을 때까지 진행
    for idx, eigenvalue in enumerate(sorted_eigenvalues):
        current_sum += eigenvalue
        num_principal_components += 1
        if current_sum / total_eigenvalues_sum >= K:
            break

    # 3. 주성분만큼 고유벡터 구하기 (eigenvectors_NxM의 열의 수를 근사값 수만큼 선택)
    principal_eigenvectors = eigenvectors_NxM[:, :num_principal_components]
    return num_principal_components, principal_eigenvectors     # 4. 선택한 주성분의 개수와 주성분 고유벡터 반환

def calculate_transformation_matrix(principal_eigenvectors):
    return principal_eigenvectors.copy()    # 1. 주성분 고유벡터를 복사하여 변환 행렬 생성

# 차분 이미지 배열와 변환 행렬을 곱하여 특성 벡터를 계산한 다음, 그 결과를 반환
def calculate_feature_vectors(diff, V):
    feature_vectors = np.dot(diff, V)   # 1. 차분 이미지 배열와 변환 행렬을 곱하여 특성 벡터 계산
    return feature_vectors              # 2. 계산된 특성 벡터를 반환


# test 단계 --------------------------------------------

def preprocessing_test(test_no):
    fname = f"face_img/test/test{test_no:03d}.jpg"              # 1. 310개의 train_xxx.jpg 라벨링하기
    image = cv2.imread(fname, cv2.IMREAD_COLOR)                 # 2. 컬러 모드로 이미지 파일 읽기
    resized_image = cv2.resize(image, (120, 150), cv2.INTER_LINEAR)               # 3. 120x150 사이즈로 픽셀 조정
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)# 4. 그레이 스케일로 변환
    histeq_image = cv2.equalizeHist(gray_image)
    # cv2.imshow("adjusted_image", adjusted_image)
    # cv2.waitKey(0)
    float_image = histeq_image.astype(np.float32)                 # 데이터 유형을 float32로 변환
    normalized_image = float_image / 255                        # 6. 픽셀 값을 나누어 정규화
    return normalized_image

# 테스트 이미지 개수(num_images)를 사용하여 전처리된 테스트 데이터를 생성하고, 그 결과를 반환
def create_test_data(num_images):
    test_data = np.empty((num_images, 18000))           # 1. 미리 (num_images, 18000) 크기의 빈 배열 생성

    # 2. 테스트 데이터의 수 만큼 반복하며 전처리 후 컬럼 형태로 변환하여 배열에 채워 넣음
    for test_no in range(num_images):
        preprocessed_image = preprocessing_test(test_no)        # 2.1 테스트 이미지 번호로 전처리된 이미지 가져오기
        image_vector = preprocessed_image.flatten()             # 2.2 전처리된 이미지를 18000x1 형태의 벡터로 변환
        test_data[test_no] = image_vector                       # 2.3 변환된 이미지 벡터를 테스트 데이터 배열에 저장
    return test_data                                            # 3. 완성된 테스트 데이터 배열 반환

# 이미지 배열(img)을 원래 형태(150x120)로 변환한 후, 그 변환된 이미지를 화면에 출력
def display_image(img):
    img_with_label = np.reshape(img, (150, 120))    # 1. 주어진 이미지 배열을 원래 형태(150x120)로 변환
    cv2.imshow("display_image",img_with_label)      # 2. 변환된 이미지를 화면에 출력

# 두 벡터(v1, v2) 사이의 유클리디안 거리를 계산한 후, 그 결과를 반환
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)  # 1. 주어진 두 벡터(v1, v2) 사이의 유클리디안 거리를 계산하고 반환

# 테스트 이미지의 특징 벡터와 학습 이미지의 특징 벡터를 입력으로 받아 가장 가까운 학습 이미지를 찾는 기능
def find_closest_train_image(test_feature_vector, train_feature_vectors):
    # 1. 테스트 이미지 특징 벡터와 모든 학습 이미지 특징 벡터 사이의 유클리드 거리 계산
    distances = np.linalg.norm(train_feature_vectors - test_feature_vector, axis=1)

    closest_image_idx = np.argmin(distances)            # 2. 테스트 이미지 특징 벡터와 가장 가까운 학습 이미지 특징 벡터의 인덱스 찾기
    min_distance = distances[closest_image_idx]         # 3. 가장 작은 거리 값을 가져옵니다.
    return closest_image_idx, min_distance              # 4. 가장 작은 거리 값, 특징 벡터의 인덱스 반환

def show_side_by_side(left_img, right_img, left_label, right_label, window_name='Result'):
    left_reshape_img = np.reshape(left_img, (150, 120))
    right_reshape_img = np.reshape(right_img, (150, 120))

    # 원본 이미지의 크기를 각각 3배로 늘립니다.
    resized_left_img = cv2.resize(left_reshape_img, (120 * 3, 150 * 3))
    resized_right_img = cv2.resize(right_reshape_img, (120 * 3, 150 * 3))

    # 두 이미지를 가로로 연결
    combined_img = np.hstack((resized_left_img, resized_right_img))

    # 각 이미지에 레이블을 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (255, 255, 255)

    # 글씨 추가하기
    cv2.putText(combined_img, left_label, (10, 40), font, font_scale, font_color, 1, cv2.LINE_AA)
    cv2.putText(combined_img, right_label, (120*3 + 10, 40), font, font_scale, font_color, 1, cv2.LINE_AA)

    # 연결된 이미지를 출력
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("train start")
    # 얼굴 데이터 처리
    num_images = 310
    train_data = create_train_data(num_images)
    print("train_data : ",train_data.shape)

    # 평균 얼굴 구하기
    mean_image = faces_avg(train_data)
    print("mean_image : ", mean_image.shape)
    # 평균 얼굴 출력 함수
    faces_avg_show(mean_image)

    # 차 백터 구하는 함수
    differences = store_difference_images(train_data, mean_image)
    print("difference_images : ", differences.shape)
    # 차 백터 출력 함수
    #show_difference_images(train_data, mean_image)

    # 벡터 X 구하기 18000x310
    train_flat_diff = flat_difference_vectors(differences)
    print("train_flat_diff : ", train_flat_diff.shape)

    # 공분산 행렬 구하기 310x310
    cov_matrix = covariance_matrix(train_flat_diff.T)
    print("cov_matirx : ",cov_matrix.shape)

    # 공분산 행렬의 고유백터와 고유값 구하기
    eigenvalues, eigenvectors = eigenvectors_and_eigenvalues(cov_matrix)
    print("eigenvalues.shape : ", eigenvalues.shape)
    print("eigenvectors.shape : ", eigenvectors.shape)

    # 내림차순으로 출력
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # print("sorted_eigenvectors", sorted_eigenvectors)
    # print("sorted_eigenvalues", sorted_eigenvalues)

    graph_plot(sorted_eigenvalues)

    # N 차원 고유 백터
    eigenvectors_NxM = compute_eigenvectors_in_original_space(train_flat_diff, eigenvectors)
    print("eigenvectors_NxM.shape : ", eigenvectors_NxM.shape)
    # test : 고유 얼굴들 출력
    #display_eigenfaces(eigenvectors_NxM)

    # K = 0.95 고유백터 선택
    num_principal_components, principal_eigenvectors = find_principal_components(sorted_eigenvalues, eigenvectors_NxM, K=0.88)
    print("Num. Principal Components:" ,num_principal_components)
    print("principal_eigenvectors.shape : ", principal_eigenvectors.shape)

    # 주요 고유 벡터를 사용하여 변환 행렬을 만듭니다.
    V = calculate_transformation_matrix(principal_eigenvectors)
    print("V shape:", V.shape)

    # 변환행렬 V와 전체 평균 영상의 차를 곱하여 특징백터를 구한다.
    feature_vectors = calculate_feature_vectors(differences, V)
    print("Feature Vectors shape:", feature_vectors.shape)

    print("train end")
    print("")

    # test 단계 --------------------------------------------

    print("")
    print("test start")

    # 테스트 데이터 처리
    num_test_images = 93
    test_data = create_test_data(num_test_images)
    print("test_data : ",test_data.shape)

    # 테스트 이미지 번호 입력 받기
    # test_image_idx = int(input("test 입력 영상 번호 입력(0~92): "))

    # 입력영상 - 평균영상
    diff_test = store_difference_images(test_data, mean_image)
    print("diff_test : ",diff_test.shape)

    # 특징값 = 차 테스트 영상 x 변환행렬
    test_feature_vectors = calculate_feature_vectors(diff_test, V)
    print("test_feature_vectors : ",test_feature_vectors.shape)

    # 테스트 특징 백터와 가장 가까운 학습 특징 벡터 찾기
    test_image_idx = int(input("test 이미지 인덱스 입력(0~92): "))
    test_feature_vector = test_feature_vectors[test_image_idx]
    closest_train_image_idx, min_distance = find_closest_train_image(test_feature_vector, feature_vectors)
    print("가장 가까운 이미지 인덱스:", closest_train_image_idx, "거리:", min_distance)

    # 테스트 결과 출력
    show_side_by_side(test_data[test_image_idx], train_data[closest_train_image_idx],
                      "Test #{}".format(test_image_idx), "Train #{}".format(closest_train_image_idx))

if __name__ == "__main__":
    main()
    cv2.waitKey(0)