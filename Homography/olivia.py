import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv

# social
def f1(x):
    if x == "Vic":
        return 29
    if x == "Sampson":
        return 27
    if x == "Shane":
        return 24
    if x == "Eason":
        return 31
    if x == "Hubert":
        return 26
    if x == "Lin":
        return 27
    if x == "Elain":
        return 34
    if x == "Jack":
        return 37
    if x == "Sophy":
        return 32
    if x == "Zack":
        return 31

# self-con
def f2(x):
    if x == "Vic":
        return 76
    if x == "Sampson":
        return 83
    if x == "Shane":
        return 71
    if x == "Eason":
        return 73
    if x == "Hubert":
        return 71
    if x == "Lin":
        return 78
    if x == "Elain":
        return 75
    if x == "Jack":
        return 79
    if x == "Sophy":
        return 79
    if x == "Zack":
        return 71

# Author
def f3(x):
    if x == "Vic":
        return 80
    if x == "Sampson":
        return 76
    if x == "Shane":
        return 70
    if x == "Eason":
        return 78
    if x == "Hubert":
        return 83
    if x == "Lin":
        return 81
    if x == "Elain":
        return 77
    if x == "Jack":
        return 86
    if x == "Sophy":
        return 76
    if x == "Zack":
        return 73

# Esteen
def f4(x):
    if x == "Vic":
        return 42
    if x == "Sampson":
        return 42
    if x == "Shane":
        return 44
    if x == "Eason":
        return 45
    if x == "Hubert":
        return 43
    if x == "Lin":
        return 49
    if x == "Elain":
        return 51
    if x == "Jack":
        return 47
    if x == "Sophy":
        return 53
    if x == "Zack":
        return 40

# Monitor
def f5(x):
    if x == "Vic":
        return 8
    if x == "Sampson":
        return 8
    if x == "Shane":
        return 6
    if x == "Eason":
        return 7
    if x == "Hubert":
        return 14
    if x == "Lin":
        return 11
    if x == "Elain":
        return 11
    if x == "Jack":
        return 13
    if x == "Sophy":
        return 9
    if x == "Zack":
        return 14


if __name__ == "__main__":
    x = ["Eason", "Elain", "Hubert", "Jack", "Lin", "Sampson", "Shane", "Sophy", "Vic", "Zack"]
    y = [f1(x_) for x_ in x]
    y1 = [26 for _ in range(len(x))]

    plt.plot(x, y)
    plt.plot(x, y1)
    plt.text(1, 25, "Olivia")

    avg = sum(y) / len(x)
    a = [avg for _ in range(len(x))]
    plt.plot(x, a)
    plt.text(1, avg + 2, "Average Score")

    for a, b in zip(x, y):
        plt.text(a, b+1, str(b))
    plt.title('Toal Perceived Social Skills')
    plt.ylim(0, 40)
    plt.show()

    # x = ["Eason", "Elain", "Hubert", "Lin", "Sampson", "Shane", "Vic"]
    y = [f2(x_) for x_ in x]

    y1 = [70 for _ in range(len(x))]

    plt.plot(x, y)
    plt.plot(x, y1)
    plt.text(1, 80, "Olivia")

    avg = sum(y) / len(x)
    a = [avg for _ in range(len(x))]
    plt.plot(x, a)
    plt.text(1, avg + 2, "Average Score")

    for a, b in zip(x, y):
        plt.text(a, b+1, str(b))
    plt.title('Grand Total Perceived Self Control')
    plt.ylim(0, 96)
    plt.show()

    # x = ["Eason", "Elain", "Hubert", "Lin", "Sampson", "Shane", "Vic"]
    y = [f3(x_) for x_ in x]

    y1 = [85 for _ in range(len(x))]

    plt.plot(x, y)
    plt.plot(x, y1)
    plt.text(1, 80, "Olivia")

    avg = sum(y) / len(x)
    a = [avg for _ in range(len(x))]
    plt.plot(x, a)
    plt.text(1, avg + 2, "Average Score")

    for a, b in zip(x, y):
        plt.text(a, b+1, str(b))
    plt.title('Total Perceived Attitude Towards Authority')
    plt.ylim(0, 120)
    plt.show()

    # x = ["Eason", "Elain", "Hubert", "Lin", "Sampson", "Shane", "Vic"]
    y = [f4(x_) for x_ in x]

    y1 = [29 for _ in range(len(x))]


    plt.plot(x, y)
    plt.plot(x, y1)
    plt.text(1, 28, "Olivia")

    avg = sum(y) / len(x)
    a = [avg for _ in range(len(x))]
    plt.plot(x, a)
    plt.text(1, avg + 2, "Average Score")
    for a, b in zip(x, y):
        plt.text(a, b+1, str(b))


    plt.title('Total Perceived Self Esteem')
    plt.ylim(0, 60)
    plt.show()

    # x = ["Eason", "Elain", "Hubert", "Lin", "Sampson", "Shane", "Vic"]
    y = [f5(x_) for x_ in x]

    y1 = [3 for _ in range(len(x))]


    plt.plot(x, y)
    plt.plot(x, y1)
    plt.text(1, 4, "Olivia")

    avg = sum(y) / len(x)
    a = [avg for _ in range(len(x))]
    plt.plot(x, a)
    plt.text(1, avg + 2, "Average Score")
    for a, b in zip(x, y):
        plt.text(a, b+1, str(b))
    plt.title('Total Perceived Self-Monitoring')
    plt.ylim(0, 18)
    plt.show()


