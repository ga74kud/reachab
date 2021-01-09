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
        self.zonotype= {'c':
                        np.array([[0],
                         [0],
                         [0],
                         [0]
                        ]),
                        'g':
                        [
                            np.array([[1], [0], [0], [0]]),
                            np.array([[0], [1], [0], [0]]),
                            np.array([[0.0], [.2], [0], [0]]),
                        ]}
        self.params={'T': 2.0,
                     'N': 10.0,
                     'gamma': 2.0 #threshold for control input constraint (inf-norm)
                     }
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
    def multiplication_on_center(self, mat):
        return mat*np.matrix(self.zonotype['c'])

    def multiplication_on_generator(self, mat, list):
        return mat*np.matrix(list)



    def compute_zonoset(self):
        x_vec = self.zonotype['c']
        for wlt in self.zonotype['g']:
            a=np.array(wlt)
            x_pos = [i+a for i in x_vec]
            x_neg = [i-a for i in x_vec]
            x_vec=x_pos+x_neg
        unique_vec=np.squeeze(np.unique(x_vec, axis=0))
        points = self.compute_convex_hull(unique_vec[:, 0], unique_vec[:, 1])
        return [points[:, 0], points[:, 1]]

    def compute_convex_hull(self, x, y):
        v=np.transpose(np.vstack([x, y]))
        hull = scipy.spatial.ConvexHull(v)
        return hull.points[hull.vertices]

    def approximate_reachable_set(self):
        '''
        algorithm from: Girard, A.; "Reachability of Uncertain Linear Systems
        Using Zonotopes"
        '''
        # 1. step
        #see self.params['N']
        inf_norm_A=np.linalg.norm(self.A, np.inf)
        r_norm_A=self.params['r']*inf_norm_A
        exp_r_norm_A=np.exp(r_norm_A)
        # 2. step
        alpha_r=(exp_r_norm_A-1-r_norm_A)
        # 3. step
        beta_r=(exp_r_norm_A-1)*self.params['gamma']/inf_norm_A
        # 4. step
        P_0={'c': None, 'g': None}
        P_0['c']=(self.zonotype['c']+self.multiplication_on_center(exp_r_norm_A))/2
        a = [0.5*(i+self.multiplication_on_generator(exp_r_norm_A, i)) for i in self.zonotype['g']]
        b = [0.5 * (i + self.multiplication_on_generator(exp_r_norm_A, i)) for i in self.zonotype['g']]
        P_0['g']=a+b
        None

if __name__ == '__main__':
    obj_reach = reachability()
    obj_visual = visualizer()
    obj_visual.show_point(obj_reach.zonotype['c'])
    zonoset=obj_reach.compute_zonoset()
    test=obj_reach.approximate_reachable_set()
    obj_visual.filled_polygon(zonoset)

    obj_visual.show()