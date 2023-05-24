import os
import cv2
import numpy as np

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
    return data_set

# 얼굴 평균 리턴
def faces_avg(faces):
    mean_vector = np.mean(faces, axis=0)
    average_img_reshaped = np.reshape(mean_vector, (-1, 120))

    return average_img_reshaped

def difference_vectors(train_faces, train_facesAvg):
    # facesSub = []
    # for face in train_faces:
    #     facesSub.append(face-train_facesAvg)

    avg_face_vector = train_facesAvg.flatten()

    diff_vectors = [face_image - avg_face_vector for face_image in train_faces]


    return diff_vectors

train_faces = load_faces(200)
train_facesAvg = faces_avg(train_faces)

print(train_facesAvg.shape)
cv2.imshow("avgFace", train_facesAvg)

train_diff = difference_vectors(train_faces, train_facesAvg)
cv2.imshow("train_faces", train_diff)

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
