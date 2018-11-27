from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
from facenet import load_dataset, load_facenet, img_to_encoding
import cv2 as cv
import numpy as np
import pickle
import os
import sklearn
from sklearn.cluster import KMeans

def q1_c():
    model = load_facenet()
    files = os.listdir("saved_faces")
    data = []
    paths = []
    for path in files:
        image = cv.imread("saved_faces/" + path)
        v = img_to_encoding(image, model)
        data.append(v)
        paths.append(path)

    with open("saved_data.pickle", "wb") as handle:
        pickle.dump([data, paths], handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_input_face():
    model = load_facenet()
    files = os.listdir("input_faces")
    data = []
    paths = []
    for path in files:
        image = cv.imread("input_faces/" + path)
        image = cv.resize(image, (96, 96))
        v = img_to_encoding(image, model)
        data.append(v)
        paths.append(path)

    with open("input_data.pickle", "wb") as handle:
        pickle.dump([data, paths], handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# load_dataset()



def load_pickle_data_saved():
    with open("saved_data.pickle", "rb") as handle:
        Data = pickle.load(handle)
        names = list(Data[1])
        embeddings = Data[0]
        print(len(embeddings))
        print(embeddings)
    return names, embeddings

def load_pickle_data_input():
    with open("input_data.pickle", "rb") as handle:
        Data = pickle.load(handle)
        names = list(Data[1])
        embeddings = Data[0]
    return names, embeddings


def cluster():
    # use sklearn's kmeans to cluster
    # Q1 e
    names, embeddings = load_pickle_data_saved()
    pre_clustering = []
    for vector in embeddings:
        pre_clustering.append(vector[0])
    kmeans = KMeans(n_clusters=6, random_state=0).fit(pre_clustering)
    # print(len(kmeans.labels_))
    print(kmeans.labels_)
    return kmeans, names

def inverted_index():
    # Q1 f
    kmeans, names = cluster()
    labels_list = kmeans.labels_
    d = {1: [], 2: [], 3: [], 4: [], 5: [], 0: []}
    for i in range(len(names)):
        d[labels_list[i]].append(names[i])
    return d, kmeans.cluster_centers_


def match_image():
    # Q1 g and h
    inverted_index_dictionary, centers = inverted_index()
    names, embeddings = load_pickle_data_input()

    images_scores = {}
    for i in range(len(names)):
        # compute norm-dot_product
        q = embeddings[i]
        scores = []
        for center_index in range(len(centers)):
            t = centers[center_index]
            score = np.divide(np.dot(q, t), (np.linalg.norm(q) * np.linalg.norm(t)))
            scores.append([score, center_index])
        images_scores[names[i]] = max(scores)

    # use a threshold to keep only high-scored matchings
    threshold = 0.8
    for key in images_scores.keys():
        if images_scores[key][0] <= threshold:
            images_scores[key][1] = "None"
        else:
            images_scores[key][0] = list(inverted_index_dictionary[images_scores[key][1]])
    # for key in images_scores.keys():
    #     print("----image: " + str(key) + " belongs to class: " + str(images_scores[key][1]) + "----")
    print(images_scores)


match_image()
