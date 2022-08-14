'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''

# Referred Documentations
# https://facerecognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#face_locations
#
# https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
#
# https://docs.opencv.org/4.x/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
#
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

from helper import show_image

import cv2
import numpy as np
import os
import sys
import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
# haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# cascade = cv2.CascadeClassifier(haar)


def detect_faces(input_path: str) -> dict:
    result_list = []
    files = os.listdir(input_path)
    for fn in files:
        tmp_image = face_recognition.load_image_file(input_path + "/" + fn)
        face_locations = face_recognition.face_locations(tmp_image)
        bounding_box = [0, 0, 0, 0]
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            x, y, width, height = left, top, right - left, bottom - top
            bounding_box = [x, y, width, height]

        tmp_dict = {"iname": fn, "bbox": bounding_box}
        result_list.append(tmp_dict)

        # Draw rectangle on image
        # tx, ty, tw, th = bounding_box
        # cv2.rectangle(tmp_image, (tx, ty), (tx+tw, ty + th), (0, 255, 0), 2)
        # cv2.imshow(fn, tmp_image)
        # cv2.waitKey()
    return result_list


'''
K: number of clusters
'''


def cluster_faces(input_path: str, K: int) -> dict:
    K = int(K)
    result_list = []
    files = os.listdir(input_path)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    en = []
    face_dicts = []
    for fn in files:
        tmp_image = face_recognition.load_image_file(input_path + "/" + fn)
        faces = face_recognition.face_locations(tmp_image)
        tmp_dict = {"file_name": fn, "location": faces}

        # To draw rectangle on face location of image
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(tmp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.imshow(file_name, tmp_image)
        #     cv2.waitKey()

        face_dicts.append(tmp_dict)
        en.append(face_recognition.face_encodings(tmp_image, faces))

    '''
    Your implementation.
    '''

    ret, label, center = cv2.kmeans(np.float32(en), K, None, criteria, 10, flags)
    retVal = []
    for x in range(len(files)):
        tmp_dict = {"iname": files[x], "label": label[x][0]}
        retVal.append(tmp_dict)

    inserted = []
    clusters = [0 for x in range(K)]

    for elem in retVal:
        tmp_values = list(elem.values())
        if tmp_values[1] in inserted:
            clusters[tmp_values[1]].append(tmp_values[0])
        else:
            clusters[tmp_values[1]] = [tmp_values[0]]
            inserted.append(tmp_values[1])

    for idx in range(len(clusters)):
        tmp_dict = {"cluster_no": str(idx), "elements": clusters[idx]}
        result_list.append(tmp_dict)

    # To create clustered images
    # clusters_name = []
    # final_images = []
    # for elem in result_list:
    #     tmp_dict = list(elem.values())
    #     clusters_name.append("cluster_" + tmp_dict[0])
    #     tmp_images = []
    #     for img in tmp_dict[1]:
    #         for elem2 in face_dicts:
    #             tmp_dict2 = list(elem2.values())
    #             if tmp_dict2[0] == img:
    #                 top, right, bottom, left = tmp_dict2[1][0]
    #                 tmp_img = cv2.imread(input_path + "/" + img)[top:bottom, left:right]
    #                 tmp_images.append(tmp_img)
    #     final_images.append(tmp_images)
    # for idx in range(len(clusters_name)):
    #     tmp_img = final_images[idx][0]
    #     tmp_img = cv2.resize(tmp_img, (150, 150))
    #     tmp_img2 = final_images[idx][1]
    #     tmp_img2 = cv2.resize(tmp_img2, (150, 150))
    #     ret_img = np.concatenate((tmp_img, tmp_img2), axis=1)
    #     for idx2 in range(2, len(final_images[idx])):
    #         temp_img = final_images[idx][idx2]
    #         temp_img = cv2.resize(temp_img, (150, 150))
    #         ret_img = np.concatenate((ret_img, temp_img), axis=1)
    #     cv2.imwrite(clusters_name[idx]+".jpg", ret_img)

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
