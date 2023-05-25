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

def create_train_data(num_images):
    train_data = np.empty((num_images, 18000))  # 미리 (num_images, 18000) 크기의 빈 배열 생성

    for train_no in range(num_images):
        preprocessed_image = preprocessing(train_no)
        image_vector = preprocessed_image.flatten()     # 120x150 영상을 18000x1로 변환
        train_data[train_no] = image_vector

    return train_data

def faces_avg(faces):
    mean_image_vector = np.mean(faces, axis=0)

    return mean_image_vector

def faces_avg_show(avg_faces):
    # 평균 이미지의 최소값과 최대값 및 데이터 타입을 uint8로 변환
    average_img_reshaped = np.reshape(avg_faces, (150, 120))
    cv2.imshow('Mean Image', average_img_reshaped)


def show_difference_images(train_data, mean_image):
    num_images = train_data.shape[0]
    mean_image_reshaped = np.reshape(mean_image, (150, 120))

    for i in range(num_images):
        train_image_vector = train_data[i]
        train_image_reshaped = np.reshape(train_image_vector, (150, 120))

        difference_image = np.abs(train_image_reshaped - mean_image_reshaped)

        # 차 이미지 출력하기
        cv2.imshow(f'Difference Image {i+1}', difference_image)
        cv2.waitKey(0)

def store_difference_images(train_data, mean_image):
    num_images = train_data.shape[0]
    differences = np.empty((num_images, 18000))

    for i in range(num_images):
        train_image_vector = train_data[i]
        difference_image_vector = train_image_vector - mean_image
        differences[i] = difference_image_vector

    return differences

def flat_difference_vectors(train_diff):
    flattened_diff = train_diff.T
    return flattened_diff

def covariance_matrix(flat_diff):
    cov_matrix = flat_diff @ flat_diff.T
    return cov_matrix

def eigenvectors_and_eigenvalues(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors

def compute_eigenvectors_in_original_space(train_flat_diff, eigenvectors_MxM):
    eigenvectors_NxM = train_flat_diff @ eigenvectors_MxM
    return eigenvectors_NxM

def display_eigenfaces(eigenvectors_NxM, save=False):
    num_eigenfaces = eigenvectors_NxM.shape[1]

    for i in range(num_eigenfaces):
        eigenface_vector = eigenvectors_NxM[:, i]
        eigenface_image = np.reshape(eigenface_vector, (150, 120))

        # 명암값을 강조하여 출력하기 쉽게 만듭니다.
        eigenface_image = cv2.normalize(eigenface_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 고유 얼굴 출력
        cv2.imshow(f'Eigenface {i + 1}', eigenface_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def graph_plot(sorted_eigenvalues):
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_eigenvalues)
    plt.title('Sorted Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.xlim(0, 400)
    plt.grid(True)
    plt.show()

def find_principal_components(sorted_eigenvalues, eigenvectors_NxM, K=0.95):
    total_eigenvalues_sum = np.sum(sorted_eigenvalues)
    current_sum = 0
    num_principal_components = 0

    for idx, eigenvalue in enumerate(sorted_eigenvalues):
        current_sum += eigenvalue
        num_principal_components += 1
        if current_sum / total_eigenvalues_sum >= K:
            break

    principal_eigenvectors = eigenvectors_NxM[:, :num_principal_components]
    return num_principal_components, principal_eigenvectors

def calculate_transformation_matrix(principal_eigenvectors):
    return principal_eigenvectors.copy()

def calculate_feature_vectors(diff, V):
    feature_vectors = np.dot(diff, V)  # V.T를 직접 사용
    return feature_vectors



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
    graph_plot(sorted_eigenvalues)

    # N 차원 고유 백터
    eigenvectors_NxM = compute_eigenvectors_in_original_space(train_flat_diff, eigenvectors)
    print("eigenvectors_NxM.shape : ", eigenvectors_NxM.shape)
    # test : 고유 얼굴들 출력
    #display_eigenfaces(eigenvectors_NxM)

    # K = 0.95 고유백터 선택
    num_principal_components, principal_eigenvectors = find_principal_components(sorted_eigenvalues, eigenvectors_NxM, K=0.95)
    print("Num. Principal Components:" ,num_principal_components)
    print("principal_eigenvectors.shape : ", principal_eigenvectors.shape)

    # 주요 고유 벡터를 사용하여 변환 행렬을 만듭니다.
    V = calculate_transformation_matrix(principal_eigenvectors)
    print("V shape:", V.shape)

    # 변환행렬 V와 전체 평균 영상의 차를 곱하여 특징백터를 구한다.
    feature_vectors = calculate_feature_vectors(differences, V)
    print("Feature Vectors shape:", feature_vectors.shape)





if __name__ == "__main__":
    main()
    cv2.waitKey(0)