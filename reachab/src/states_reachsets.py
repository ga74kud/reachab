# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2021 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

def show_points(points, color):
    plt.scatter(points[:, 0], points[:, 1], edgecolors=color)


def show_filled_polygon(points, the_color):
    plt.fill(points[:, 0], points[:, 1], color=the_color, alpha=.2)
'''
    is x inside the points 
    idea is to solve a linear program
    original code from "https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl"
'''
def inside_polygon(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def check_points_inside(points, polygon_points):
    inside_points=[]
    for wlt in range(0, np.size(points, 0)):
        act_point=points[wlt, :]
        bool_val=inside_polygon(polygon_points, act_point)
        if(bool_val):
            inside_points.append(act_point)
    inside_points=np.array(inside_points)
    return inside_points

def get_points_inside(polygon_points):
    x = polygon_points[:, 0]
    y = polygon_points[:, 1]
    xmin,xmax,ymin,ymax=np.min(x), np.max(x), np.min(y), np.max(y)
    x_lin, ylin=np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10)
    X, Y=np.meshgrid(x_lin, ylin)
    X,Y=np.ravel(X), np.ravel(Y)
    meshgrid_points=np.transpose(np.vstack((X, Y)))
    inside_points=check_points_inside(meshgrid_points, polygon_points)
    return inside_points
def get_sample_points_inside_hull(zonoset):
    all_inside_points=[]
    for act_points in zonoset:
        points=np.transpose(np.vstack((act_points[0], act_points[1])))
        inside_points=get_points_inside(points)
        all_inside_points.append(inside_points)
    return all_inside_points


if __name__ == '__main__':
    n_points = 10000
    n_dim = 2
    Z = np.array([[0, 5], [4, 2], [8, 4], [8, 5], [6, 8], [1, 6]])
    x = np.array([[4, 5]])



    for act_point in x:
        print(inside_polygon(Z, act_point))

    show_filled_polygon(Z, 'green')
    show_points(Z, 'blue')
    show_points(x, 'red')
    plt.show()