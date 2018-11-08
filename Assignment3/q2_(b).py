# For affine transformation we need at least 3 points
# For homography transformation we need at least 4 points
# P is 0.99, p is visually estimated to 0.8, 0.72 and 0.66
import numpy as np

def Q2_b_affine(P, p, k):
    return (np.log(1 - P)) / (np.log(1 - pow(p, k)))

def Q2_b_homography(P, p, k):
    return (np.log(1 - P)) / (np.log(1 - pow(p, k)))

print("Iterations needed for affine")
print(Q2_b_affine(0.99, 0.88, 3))
print(Q2_b_affine(0.99, 0.83, 3))
print(Q2_b_affine(0.99, 0.47, 3))

print("-------------------------------")

print("Iterations needed for homography")
print(Q2_b_affine(0.99, 0.88, 4))
print(Q2_b_affine(0.99, 0.83, 4))
print(Q2_b_affine(0.99, 0.47, 4))
