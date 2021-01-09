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
                        ]}
        self.params={'T': 2.0,
                     'N': 4,
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
    def convert_to_matrix(self, list):
        erg=np.hstack((list[0], list[1]))
        for i in range(2, len(list)):
            erg=np.hstack((erg, list[i]))
        return erg
    def multiplication_on_zonotype(self, mat, zonotype):
        Z = {'c': None, 'g': None}
        Z['c']=mat*zonotype['c']
        for i in range(0, np.size(zonotype['g'], 0)):
            act_g=np.matrix(zonotype['g'][i,:])
            if(i==0):
                g=mat*np.transpose(act_g)
            else:
                g=np.hstack((g, mat*np.transpose(act_g)))
        Z['g']=np.transpose(np.array(g))
        return Z

    def get_unique_vectors(self, vec):
        unique_vec = np.squeeze(np.unique(vec, axis=0))
        return unique_vec
    def compute_zonoset(self, c, g):
        x_vec = c
        for wlt in g:
            a=np.array(wlt)
            x_pos = [i+a for i in x_vec]
            x_neg = [i-a for i in x_vec]
            x_vec=x_pos+x_neg
        unique_vec=self.get_unique_vectors(x_vec)
        points = self.compute_convex_hull(unique_vec[:, 0], unique_vec[:, 1])
        return [points[:, 0], points[:, 1]]


    def compute_convex_hull(self, x, y):
        v=np.transpose(np.vstack([x, y]))
        hull = scipy.spatial.ConvexHull(v)
        return hull.points[hull.vertices]

    def minkowski_zonotypes(self, ZA, ZB):
        Z={'c': None, 'g': None}
        Z['c']=ZA['c']+ZB['c']
        a=np.array(ZA['g'])
        b=np.array(ZB['g'])
        new_g=np.vstack((a, b))
        Z['g'] = new_g
        return Z
    def square_zonotype(self, radius):
        Z={'c': np.array([[0],
                         [0],
                         [0],
                         [0]
                        ]),
           'g': None
           }
        b=[
               np.array([[radius], [0], [0], [0]]),
               np.array([[0], [radius], [0], [0]]),
               np.array([[0], [0], [radius], [0]]),
               np.array([[0], [0], [0], [radius]]),
           ]
        Z['g']=self.get_unique_vectors(b)
        return Z

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
        exp_r_A = np.exp(self.params['r']*self.A)
        # 2. step
        alpha_r=(exp_r_norm_A-1-r_norm_A)
        # 3. step
        beta_r=(exp_r_norm_A-1)*self.params['gamma']/inf_norm_A
        # 4. step
        P_0={'c': None, 'g': None}
        P_0['c']=np.array((self.zonotype['c']+self.multiplication_on_center(exp_r_norm_A))/2)
        a = [0.5*(i+self.multiplication_on_generator(exp_r_norm_A, i)) for i in self.zonotype['g']]
        b = [0.5 * (i + self.multiplication_on_generator(exp_r_norm_A, i)) for i in self.zonotype['g']]
        P_0['g']=self.get_unique_vectors(a+b)
        # 5. step
        rad=alpha_r+beta_r
        square_Z=self.square_zonotype(rad)
        Q_0=self.minkowski_zonotypes(P_0, square_Z)
        # 6. step
        all_R=[]
        Q_i = Q_0
        R_i=Q_0
        all_R.append(R_i)
        for i in range(1,self.params['N']-1):
            # 7. step
            P_i=self.multiplication_on_zonotype(exp_r_norm_A, Q_i)
            # 8. step
            square_Z = self.square_zonotype(beta_r)
            Q_i = self.minkowski_zonotypes(P_i, square_Z)
            # 9. step
            all_R.append(Q_i)
        return all_R

if __name__ == '__main__':
    obj_reach = reachability()
    obj_visual = visualizer()
    obj_visual.show_point(obj_reach.zonotype['c'])
    zonoset_init=obj_reach.compute_zonoset(obj_reach.zonotype['c'], obj_reach.zonotype['g'])
    R=obj_reach.approximate_reachable_set()
    obj_visual.filled_polygon(zonoset_init, 'lightsalmon')
    for act_zono in R:
        zonoset_P0 = obj_reach.compute_zonoset(act_zono['c'], act_zono['g'])
        obj_visual.filled_polygon(zonoset_P0, 'green')
    obj_visual.show()