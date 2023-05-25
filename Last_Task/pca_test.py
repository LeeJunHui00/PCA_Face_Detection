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


train_faces = load_faces(310)
print("train_faces.shape : ", train_faces.shape)
train_facesAvg = faces_avg(train_faces)

# 평균치 영상 사이즈 체크
print("train_facesAvg.shape : ",train_facesAvg.shape)
# 평균치 얼굴 출력
cv2.imshow("mean_image", train_facesAvg)

train_diff = difference_vectors(train_faces, train_facesAvg)
print("train_diff.shape : ",train_diff.shape)

# flat_diff = flat_difference_vectors(train_diff)
# print("flat_diff : ",flat_diff.shape)

# -----------------------------------

# reshaped_flattened_diff = np.reshape(flat_diff, (150, 120))
# print("reshaped_flattened_diff : ",reshaped_flattened_diff.shape)

# 평탄화된 차 영상 출력
# cv2.imshow("Flattened Difference Vectors", reshaped_flattened_diff)

# -----------------------------------


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

K = 0.95
selected_eigenvalues_by_k, selected_eigenvectors_by_k =select_eigenvectors_by_k(sorted_eigenvalues, sorted_eigenvectors, K)
print("selected_eigenvalues_by_k shape:", selected_eigenvalues_by_k.shape)
print("selected_eigenvectors_by_k shape:", selected_eigenvectors_by_k.shape)

graph_plot(sorted_eigenvalues)



# 차 영상 출력
# show_difference_vectors(train_diff)


# print(load_faces(200))

# X = np.array(data)
# X = np.float32(X)

# 영상의 평균 벡터를 구합니다.
# mean_vector = np.mean(X, axis=0)

# 평균 벡터를 2차원 영상(n * m)으로 변환합니다.
# mean_image = np.reshape(mean_vector, (120, 150))
#
#
# cv2.imshow("Mean Image", mean_image)
# print(mean_image)


# 사용자가 키 입력을 기다립니다.
cv2.waitKey(0)

# 사용자가 키를 누르면 모든 창을 닫습니다.
cv2.destroyAllWindows()
