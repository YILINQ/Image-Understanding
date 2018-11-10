import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

f = 721.537700
px = 609.559300
py = 172.854000
baseline = 0.5327119288

def process_data(filename):
    d = {}
    fileIO = open(filename, "r")
    count = 0
    num_detection = int(fileIO.readline().strip())
    scores_list = [float(x) for x in fileIO.readline().strip("[]\n").split(",")]
    classes = [int(x) for x in fileIO.readline().strip("[]\n ").split(",")]
    boxes = fileIO.readline().strip("\n]")[1:-1].split("],")
    detection_boxes = []
    for item in boxes:
        item_list = item.strip()[1:].split(", ")
        coor_list = [float(x) for x in item_list]
        detection_boxes.append(coor_list)
    d["num_detections"] = num_detection
    d["detection_boxes"] = detection_boxes
    d["detection_scores"] = scores_list
    d["detection_classes"] = classes
    fileIO.close()
    return d

detection1, detection2, detection3 = process_data("data/test/q2b_4945.txt"), process_data("data/test/q2b_4964.txt"), process_data("data/test/q2b_5002.txt")
img1 = cv.imread("data/test/left/004945.jpg")
img2 = cv.imread("data/test/left/004964.jpg")
img3 = cv.imread("data/test/left/005002.jpg")

def q2_a():
    # first image
    D1 = cv.imread("data/test/results/004945_left_disparity.png")
    f1 = 721.537700
    px1 = 609.559300
    py1 = 172.854000
    baseline1 = 0.5327119288

    result1 = np.zeros(D1.shape)

    result1 = np.true_divide((f1 * baseline1), D1)
    cv.imwrite("4945_result.png", result1)

    # second image
    D2 = cv.imread("data/test/results/004964_left_disparity.png")
    f2 = 721.537700
    px2 = 609.559300
    py2 = 172.854000
    baseline2 = 0.5327119288

    result2 = np.zeros(D1.shape)
    result2 = np.true_divide((f2 * baseline2), D2)

    cv.imwrite("4964_result.png", result2)

    # third image
    D3 = cv.imread("data/test/results/005002_left_disparity.png")
    f3 = 721.537700
    px3 = 609.559300
    py3 = 172.854000
    baseline3 = 0.5327119288

    result3 = np.zeros(D3.shape)
    result3 = np.true_divide((f3 * baseline3), D3)

    cv.imwrite("5002_result.png", result3)
    return D1, D2, D3
D1, D2, D3 = q2_a()

def draw(img, pos, classes, confidence):
    left = int(pos[0]*img.shape[0])
    top = int(pos[1]*img.shape[1])
    right = int(pos[2]*img.shape[0])
    bot = int(pos[3]*img.shape[1])

    color = (0, 0, 0)
    text = ""

    if classes == 1:
        # person
        color = (255, 0, 0)
        text = "person:" + str(int(confidence*100)) + "%"
    if classes == 2:
        # bic
        color = (0, 255, 0)
        text = "bicycle:" + str(int(confidence*100)) + "%"
    if classes == 3:
        # car
        color = (0, 0, 255)
        text = "car:" + str(int(confidence*100)) + "%"
    if classes == 10:
        # traffic light
        color = (255, 255, 0)
        text = "trafic light:" + str(int(confidence*100)) + "%"
    cv.rectangle(img, (top, left-15), (bot, right), color=color, thickness=3)
    cv.putText(img, text, org=(top, left), color=(255, 255, 255), fontScale=1, fontFace=1)

def q2_c():
    for i in range(detection1["num_detections"]):
        draw(img1, detection1["detection_boxes"][i], detection1["detection_classes"][i], detection1["detection_scores"][i])
    cv.imwrite("q2_c_4945.jpg", img1)

    for i in range(detection2["num_detections"]):
        draw(img2, detection2["detection_boxes"][i], detection2["detection_classes"][i], detection2["detection_scores"][i])
    cv.imwrite("q2_c_4964.jpg", img2)

    for i in range(detection3["num_detections"]):
        draw(img3, detection3["detection_boxes"][i], detection3["detection_classes"][i], detection3["detection_scores"][i])
    cv.imwrite("q2_c_5002.jpg", img3)

# #------ uncomment to run Q2 c ------
q2_c()
# # ------ uncomment to run Q2 c ------
def calculate_center_of_mass(detection, D):
    # q2_d()
    centers = []
    for i in range(detection["num_detections"]):
        pos = detection["detection_boxes"][i]
        left = int(np.round(pos[0] * img1.shape[0]))
        top = int(np.round(pos[1] * img1.shape[1]))
        right = int(np.round(pos[2] * img1.shape[0]))
        bot = int(np.round(pos[3] * img1.shape[1]))
        z = np.true_divide((f * baseline), D)
        z[np.where(np.isinf(z))] = 5000
        Zs = []
        for row in range(left, right):
            for col in range(top, bot):
                X = np.true_divide((col - px)*z[row, col, 0], f)
                Y = np.true_divide((row - py)*z[row, col, 0], f)
                Zs.append((z[row, col, 0], X, Y))
        Zs.sort()
        median = Zs[len(Zs)//2]
        centers.append((median[1], median[2], median[0]))
    return centers

def calculate_3D_location(img, D):
    X, Y, Z = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)
    z = np.true_divide((f * baseline), D)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            X[row, col] = np.true_divide((col - px) * z[row, col, 0], f)
            Y[row, col] = np.true_divide((row - py) * z[row, col, 0], f)
            Z[row, col] = z[row, col, 0]
    return X, Y, Z

def q2_e(detection, D, img, out):
    X, Y, Z = calculate_3D_location(img, D)
    center = calculate_center_of_mass(detection, D)
    segment = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(detection["num_detections"]):
        pos = detection["detection_boxes"][i]
        left = int(np.round(pos[0] * img.shape[0]))
        top = int(np.round(pos[1] * img.shape[1]))
        right = int(np.round(pos[2] * img.shape[0]))
        bot = int(np.round(pos[3] * img.shape[1]))
        z = np.true_divide((f * baseline), D)
        z[np.where(z == np.inf)] = 5000
        for row in range(left, right):
            for col in range(top, bot):
                color = ((i+1)*30, 25*(i+1), 125)
                if (
                            (center[i][0] - X[row, col, 0])**2 +
                            (center[i][1] - Y[row, col, 0])**2 +
                            (center[i][2] - z[row, col, 0])**2
                )**0.5 <= 3:
                    segment[row, col] = color
    cv.imwrite(out, segment)

# #------ uncomment to run Q2 d, e ------
q2_e(detection1, D1, img1, "q2_e_004945.jpg")
q2_e(detection2, D2, img2, "q2_e_004964.jpg")
q2_e(detection3, D3, img3, "q2_e_005002.jpg")
# #------ uncomment to run Q2 d, e ------

def q2_f(detection, D):
    center = calculate_center_of_mass(detection, D)
    dis_dict = {"car": [np.inf, -1], "person": [np.inf, -1], "bicycle": [np.inf, -1]}
    num_cars, num_bics, num_people = 0, 0, 0
    is_traffic = False
    for i in range(detection["num_detections"]):
        X, Y, Z = center[i][0], center[i][1], center[i][2]
        distance = (X**2 + Y**2 + Z**2)**0.5
        if detection["detection_classes"][i] == 1:
            num_people += 1
            if distance < dis_dict["person"][0]:
                dis_dict["person"][0] = distance
                dis_dict["person"][1] = i
        if detection["detection_classes"][i] == 2:
            num_bics += 1
            if distance < dis_dict["bicycle"][0]:
                dis_dict["bicycle"][0] = distance
                dis_dict["bicycle"][1] = i
        if detection["detection_classes"][i] == 3:
            num_cars += 1
            if distance < dis_dict["car"][0]:
                dis_dict["car"][0] = distance
                dis_dict["car"][1] = i
        if detection["detection_classes"][i] == 10:
            label = "traffic light"
            dis_dict[label] = [distance, i]
            is_traffic = True
    print("There is(are) " + str(num_cars) + " car(s) nearby.")
    print("There is(are) " + str(num_people) + " persons nearby.")
    print("There is(are) " + str(num_bics) + " bicycle(s) nearby.")
    if is_traffic:
        print("Traffic light(s) near by~ Be careful!")
    if num_cars > 0:
        X = center[dis_dict["car"][1]][0]
        if X >= 0:
            text = "to your right."
        else:
            text = "to your left."
        print("The closest car is " + str(abs(X)) + " meters " + text)
        print("It is " + str(dis_dict["car"][0]) + " meters away from you.")
    if num_bics > 0:
        X = center[dis_dict["bicycle"][1]][0]
        if X >= 0:
            text = "to your right."
        else:
            text = "to your left."
        print("The closest bicycle is " + str(abs(X)) + " meters " + text)
        print("It is " + str(dis_dict["bicycle"][0]) + " meters away from you.")
    if num_people > 0:
        X = center[dis_dict["person"][1]][0]
        if X >= 0:
            text = "to your right."
        else:
            text = "to your left."
        print("The closest person is " + str(abs(X)) + " meters " + text)
        print("It is " + str(dis_dict["person"][0]) + " meters away from you.")


# #------ uncomment to run Q2 f ------
print("\n")
print("----------Info for image 004945----------")
q2_f(detection1, D1)
print("----------Info for image 004945----------")
print("\n")
print("----------Info for image 004964----------")
q2_f(detection2, D2)
print("----------Info for image 004964----------")
print("\n")
print("----------Info for image 005002----------")
q2_f(detection3, D3)
print("----------Info for image 005002----------")
# #------ uncomment to run Q2 f ------



def rent_bike(station_id, stations):
    ID = 0
    BIKES_AVALIABLE = 5
    DOCKS_AVALIABLE = 6
    for station in stations:
        if station[ID] == station_id:
            if station[BIKES_AVALIABLE] >= 1:
                station[BIKES_AVALIABLE] -= 1
                station[DOCKS_AVALIABLE] += 1
                return True
            else:
                return False
