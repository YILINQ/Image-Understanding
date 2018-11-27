from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
from facenet import load_dataset, load_facenet, img_to_encoding
import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans


def get_embedding_path():
    """
    q1(c)
    """
    model = load_facenet()
    files = os.listdir("saved_faces")
    embedding, paths = [], []
    for path in files:
        image = cv.imread("saved_faces/" + path)
        encode = img_to_encoding(image, model)
        embedding.append(encode)
        paths.append(path)

    return embedding, paths


def kmean_cluster():
    """
    q1(d)
    """
    embedding, paths = get_embedding_path()
    # store the first element of the embedding list
    cluster = []
    for v in embedding:
        cluster.append(v[0])
    kmeans = KMeans(n_clusters=6, random_state=0).fit(cluster)
    kmean_label = kmeans.labels_
    kmean_center = kmeans.cluster_centers_

    return kmean_label, kmean_center, paths


def inverted_index():
    """
    q1(f)
    """
    kmean_label, kmean_center, paths = kmean_cluster()
    inverted_idx = {}
    for i in range(len(paths)):
        index = kmean_label[i]
        if index in inverted_idx:
            inverted_idx[index].append(paths[i])
        else:
            inverted_idx[index] = [paths[i]]

    return inverted_idx


# q1(g):
# For each image in input_faces, you want to find it’s matching images from
# saved_faces. Describe a method to do this:
# 1. get the list of embeddings and paths of the input images
# 2. for each input image:
#   -> get its embedding vector
#   -> Compute the similarity by normalized dot product between its embedding
#      vector and the position of the center of each cluster, say: sim(v, c)
#   -> Let the maximum sim(v, c) as the score of current image file
# 3. In order to get rid of some mismatch, given a threshold to filter the
#    the possible mismatching
#   -> For input image with similarity <= 0.8, identify it as the 'None' type
#   -> For input image with similarity > 0.8:
#      use the inverted_index() as helper function to identify which class
#      it belongs to.
# 4. Print out the classification in order
def loading_input():
    """
    q1(h) helper function, return the (embedding, paths) of the input file
    """
    model = load_facenet()
    input_img = os.listdir("input_faces")
    embedding, paths = [], []
    for path in input_img:
        image = cv.imread("input_faces/" + path)
        image = cv.resize(image, (96, 96))
        encode = img_to_encoding(image, model)
        embedding.append(encode)
        paths.append(path)

    return embedding, paths


def matching_algo():
    """
    q1(h)
    """
    dataset_inverted_index = inverted_index()

    _, kmean_center, _ = kmean_cluster()
    input_embedding, input_paths = loading_input()

    # calculate the max sim(t, v) of each input image file
    image_sim = {}
    for i in range(len(input_paths)):
        vector = input_embedding[i]
        possible_sim = []
        for j in range(len(kmean_center)):
            center = kmean_center[j]
            sim = np.divide(np.dot(vector, center), (np.linalg.norm(vector) * np.linalg.norm(center)))
            possible_sim.append([sim, j])
        image_sim[input_paths[i]] = max(possible_sim)  # (max_sim, center)

    # threshold to filter the mismatching
    threshold = 0.8
    for name in image_sim:
        if image_sim[name][0] <= threshold:
            image_sim[name][1] = 'None'
        else:
            center_index = image_sim[name][1]
            target = dataset_inverted_index[center_index]
            image_sim[name][0] = target

    # print out the final classification
    for name in image_sim:
        print("Input image: " + name)
        print(" belongs to class: ")
        print(image_sim[name][1])
        print("==============================")

    # follow
    output = {}
    for name in image_sim:
        if image_sim[name][1] != 'None':
            category = dataset_inverted_index[image_sim[name][1]]
            output[name] = category
        else:
            output[name] = []
    print(output)


if __name__ == "__main__":
    matching_algo()
