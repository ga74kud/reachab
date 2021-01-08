# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

from util.visualizer import *
from util.util_functions import *
import numpy as np
import scipy.spatial


class reachability(object):
    def __init__(self, **kwargs):
        self.zonotype= {'c': [[2],
                              [3]],
                        'g': [[1, 0, 2.1],
                              [0, 1, 1.4]]}
        self.params={'T': 2, 'N': 10}
        self.params['r']=self.params['T']/(self.params['N']+1)
        self.system_dynamics()




    def system_dynamics(self):
        self.A = np.matrix([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.B = np.matrix([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        self.Phi=np.exp(self.params['r']*self.A)
    def multiplication_on_generator(self, mat):
        return np.matrix(self.zonotype['c'])*mat



    def compute_zonoset(self):
        x_vec = self.zonotype['c'][0]
        y_vec = self.zonotype['c'][1]
        for i in range(0, len(self.zonotype['g'][0])):
            new_x = self.zonotype['g'][0][i]
            new_y = self.zonotype['g'][1][i]
            x_pos = [i + new_x for i in x_vec]
            x_neg = [i - new_x for i in x_vec]
            x_vec=x_pos+x_neg
            y_pos = [i + new_y for i in y_vec]
            y_neg = [i - new_y for i in y_vec]
            y_vec = y_pos + y_neg
        points = self.compute_convex_hull(x_vec, y_vec)
        return [points[:, 0], points[:, 1]]

    def compute_convex_hull(self, x_vec, y_vec):
        v=np.transpose([x_vec, y_vec])
        hull = scipy.spatial.ConvexHull(v)
        return hull.points[hull.vertices]


if __name__ == '__main__':
    obj_reach = reachability()
    obj_visual = visualizer()
    obj_visual.show_point(obj_reach.zonotype['c'])
    zonoset=obj_reach.compute_zonoset()

    obj_visual.filled_polygon(zonoset)

    obj_visual.show()