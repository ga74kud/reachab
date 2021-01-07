import numpy as np

def distance_2_points(a, b):
    dist = (a[0]-b[0])**2+(a[1]-b[1])**2
    return np.sqrt(dist)