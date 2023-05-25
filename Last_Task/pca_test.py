import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(train_no):
    fname = f"face_img/train/train{train_no:03d}.jpg"
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (120, 150))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    float_image = gray_image.astype(np.float32)
    normalized_image = float_image / 255
    return normalized_image

def load_faces(num_images):
    data_set = []
    for i in range(num_images):

        image = preprocessing(i)

        # cv2.imshow("image", image)

        if image is not None:
            flatten_image = image.flatten()
            data_set.append(flatten_image)
    return np.array(data_set)

# 얼굴 평균 리턴
def faces_avg(faces):
    mean_vector = np.mean(faces, axis=0)
    average_img_reshaped = np.reshape(mean_vector, (150, 120))
    return average_img_reshaped

def difference_vectors(train_faces, train_facesAvg):
    # faceSub = []
    # for face in train_faces:
    #     faceSub.append(face-train_facesAvg)
    train_facesAvg = train_facesAvg.flatten()
    faceSub = train_faces - train_facesAvg
    return faceSub

def show_difference_vectors(faces):
    for i, face in enumerate(faces):
        reshaped_face = np.reshape(face, (150, 120))
        cv2.imshow(f"show_difference_vectors {i}", reshaped_face)
        cv2.waitKey(0)

#차 백터 구하기
def flat_difference_vectors(train_diff):
    # absolute_diff = np.abs(train_diff)
    flattened_diff = np.mean(train_diff, axis=0)
    return flattened_diff

#공분산 행렬 구하기 N*N
def covariance_matrix(train_diff):
    # flat_diff_reshaped = train_diff.reshape(1, -1)    # 전치행렬 구하기 (행렬을 전치)
    # cov_matrix = np.cov(train_diff.T)    # 공분산 행렬 구하기
    cov_matrix = train_diff @ train_diff.T
    return cov_matrix

# 고유값과 고유백터 구하기
def eigenvectors_and_eigenvalues(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors

def select_top_eigenvectors(eigenvalues, eigenvectors, v):
    sorted_indices = np.argsort(eigenvalues)[-v:][::-1]
    selected_eigenvalues = eigenvalues[sorted_indices]
    selected_eigenvectors = eigenvectors[:, sorted_indices]
    return selected_eigenvalues, selected_eigenvectors

def select_eigenvectors_by_k(eigenvalues, eigenvectors, K):
    total_sum = np.sum(eigenvalues)
    target_sum = 0
    current_sum = 0

    for i, value in enumerate(sorted_eigenvalues):
        target_sum += value
        current_sum += value
        if current_sum / total_sum >= K:
            break

    v = i + 1
    print("최소 고유값 개수 V : ", v)
    return select_top_eigenvectors(eigenvalues, eigenvectors, v)

def graph_plot(sorted_eigenvalues):
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_eigenvalues)
    plt.title('Sorted Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.xlim(0, 400)
    # plt.ylim(0, 9 * 10 ** 8)
    plt.grid(True)
    plt.show()

# 고유 얼굴(eigenfaces) 생성
def eigenfaces(selected_eigenvectors, train_diff):
    eigen_faces = train_diff.T @ selected_eigenvectors
    return eigen_faces


# 고유 얼굴(eigenfaces) 출력 함수
def show_eigenfaces(eigen_faces):
    for i, face in enumerate(eigen_faces.T):
        # 얼굴 데이터를 재구성하고 절대값 적용
        reshaped_face = np.reshape(np.abs(face), (150, 120))

        # 이미지를 0-255 범위로 정규화
        scaled_face = cv2.normalize(reshaped_face, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imshow(f"Eigenface {i}", scaled_face)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_vectors(eigenvectors, train_diff):
    difference_vectors = []
    for face_image in eigenvectors:
        face_image = face_image.reshape(-1,1)
        difference_vectors.append(eigenvectors @ face_image)

    return np.array(difference_vectors)


# test 처리하기 -----------------------------------
def preprocessing_test(test_no):
    fname = f"face_img/test/test{test_no:03d}.jpg"
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (120, 150))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    float_image = gray_image.astype(np.float32)
    normalized_image = float_image / 255
    return normalized_image

def load_faces_test(num_images):
    data_set = []
    for i in range(num_images):

        image = preprocessing_test(i)

        # cv2.imshow("image", image)

        if image is not None:
            flatten_image = image.flatten()
            data_set.append(flatten_image)
    return np.array(data_set)

def project_onto_eigenspace(eigenvectors, matrix):
    return np.dot(eigenvectors.T, matrix)

train_faces = load_faces(310)
print("train_faces.shape : ", train_faces.shape)
train_facesAvg = faces_avg(train_faces)

# 평균치 영상 사이즈 체크
print("train_facesAvg.shape : ",train_facesAvg.shape)
# 평균치 얼굴 출력
cv2.imshow("mean_image", train_facesAvg)

train_diff = difference_vectors(train_faces, train_facesAvg)
print("train_diff.shape : ",train_diff.shape)


cov_matrix = covariance_matrix(train_diff)
# 공분산 행렬 출력
print("Covariance Matrix shape:", cov_matrix.shape)
# print("")
#
eigenvalues, eigenvectors = eigenvectors_and_eigenvalues(cov_matrix)
print("eigenvalues.shape : " , eigenvalues.shape)
print("eigenvectors.shape : ", eigenvectors.shape)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# 그래프 출력
graph_plot(sorted_eigenvalues)


K = 0.95
selected_eigenvalues_by_k, selected_eigenvectors_by_k =select_eigenvectors_by_k(sorted_eigenvalues, sorted_eigenvectors, K)
print("selected_eigenvalues_by_k shape:", selected_eigenvalues_by_k.shape)
print("selected_eigenvectors_by_k shape:", selected_eigenvectors_by_k.shape)


# 고유 얼굴 생성과 출력
eigen_faces = eigenfaces(selected_eigenvectors_by_k, train_diff)
print("eigen_faces : ", eigen_faces.shape)
# show_eigenfaces(eigen_faces)

trainVecs = feature_vectors(selected_eigenvectors_by_k, train_diff)
print("trainVecs : ", trainVecs.shape)

reduced_feature_vectors = np.dot(train_diff, eigen_faces)
print("reduced_feature_vectors : ",reduced_feature_vectors.shape)



# test --------------
# test_faces = load_faces_test(93)
# print("test_faces : ",test_faces.shape)
#
# test_facesSub = difference_vectors(test_faces, train_facesAvg)
# print("test_facesSub : ", test_facesSub.shape)
#
# testVecs = project_onto_eigenspace(selected_eigenvectors_by_k, test_facesSub)
# print("testVecs : ", testVecs.shape)

# 사용자가 키 입력을 기다립니다.
cv2.waitKey(0)

# 사용자가 키를 누르면 모든 창을 닫습니다.
cv2.destroyAllWindows()
