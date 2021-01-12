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
from scipy import signal

class reachability(object):
    def __init__(self, **kwargs):

        self.params={'T': 1.2,
                     'N': 10,
                     'gamma': 0.01 #threshold for control input constraint (inf-norm)
                     }
        self.params['r']=self.params['T']/(self.params['N']+1)
        self.system_dynamics()




    def system_dynamics(self):
        A = np.matrix([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        B = np.matrix([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        C = np.eye(4)
        D=np.zeros((4, 2))
        self.discrete_sys=signal.StateSpace(A, B, C, D, dt=self.params['r'])
        self.A=np.array(self.discrete_sys.A)
        self.B = np.array(self.discrete_sys.B)
        #see Otto Föllinger, "Regelungstechnik" and Thesis of Matthias Althoff:
        self.Phi=np.eye(4)+self.params['r']*self.A+1/(2)*(A*self.params['r'])**2+1/(6)*(A*self.params['r'])**3+1/(24)*(A*self.params['r'])**4

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
        for i in range(0, np.size(zonotype['g'], 1)):
            act_g=zonotype['g'][:,i]
            if(i==0):
                g=mat*act_g
            else:
                g=np.hstack((g, mat*act_g))
        Z['g']=g
        return Z

    def get_unique_vectors(self, vec):
        unique_vec = np.squeeze(np.unique(vec, axis=0))
        return unique_vec
    def compute_zonoset(self, c, g):
        x_vec = [c]
        for wlt in range(0, np.size(g,1)):
            a=g[:, wlt]
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
        a=ZA['g']
        b=ZB['g']
        new_g=np.hstack((a, b))
        Z['g'] = new_g
        return Z
    def square_zonotype(self, radius):
        Z={'c': np.array([[0],
                         [0],
                         [0],
                         [0],
                        [0]
                        ]),
           'g': None
           }
        b=[
               np.array([[radius], [0], [0], [0], [0]]),
               np.array([[0], [radius], [0], [0], [0]]),
               np.array([[0], [0], [radius], [0], [0]]),
               np.array([[0], [0], [0], [radius], [0]]),
               np.array([[0], [0], [0], [0], [radius]]),
           ]
        Z['g']=self.get_unique_vectors(b)
        return Z

    def approximate_reachable_set_v2(self):
        all_R = []
        all_X = []
        '''
        algorithm from: Girard, A.; "Efficient Computation of Reachable Sets of Linear Time-Invariant Systems with Inputs"
        '''
        # input
        Omega_0 = {'c': np.matrix([[0],
                            [0],
                            [0],
                            [0]
                            ]),
             'g': np.matrix([[1, -1],
                             [1, 1],
                             [0, 0],
                             [0, 0]
                            ])
             }
        all_R.append(Omega_0)
        all_X.append(Omega_0)
        U = {'c': np.matrix([[0],
                             [0],
                             [0],
                             [0],
                                   ]),
                   'g': np.matrix([[1, 0],
                                   [0, 1],
                                   [0, 0],
                                   [0, 0]
                                   ])
                   }
        #self.params['N']

        # 1. step
        X_0=Omega_0
        X_i=X_0
        # 2. step
        V_0=U
        V_i=V_0
        # 3. step
        S_0={'c': np.matrix([[0],
                             [0],
                             [0],
                             [0],
                                   ]),
                   'g': np.matrix([[0, 0],
                                   [0, 0],
                                   [0, 0],
                                   [0, 0]
                                   ])
                   }
        S_i=S_0
        # 4. step
        for i in range(0, self.params['N'] - 1):
            # 5. step
            X_i = self.multiplication_on_zonotype(self.Phi, X_i)
            all_X.append(X_i)
            # 6. step
            S_i=self.minkowski_zonotypes(S_i, V_i)
            # 7. step
            V_i = self.multiplication_on_zonotype(self.Phi, V_i)
            # 8. step
            Omega_i=self.minkowski_zonotypes(X_i, S_i)
            all_R.append(Omega_i)
        return all_R, all_X
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

    def sampling_trajectory(self, x0):
        xk=x0
        uk=np.matrix([[self.params['gamma']], [self.params['gamma']]])
        for i in range(0, self.params['N']):
            xnew=self.params['r']*(self.A*xk[:, -1]+self.B*uk)+xk[:, -1]
            xk=np.append(xk, xnew, axis=1)
        traj=[(xk[0,i], xk[1,i]) for i in range(np.size(xk, 1))]
        return traj

if __name__ == '__main__':
    obj_reach = reachability()
    obj_visual = visualizer()
    #obj_visual.show_point(obj_reach.zonotype['c'])
    #zonoset_init=obj_reach.compute_zonoset(obj_reach.zonotype['c'], obj_reach.zonotype['g'])
    #R=obj_reach.approximate_reachable_set()
    R, X=obj_reach.approximate_reachable_set_v2()
    #obj_visual.filled_polygon(zonoset_init, 'lightsalmon')
    for act_zono in R:
        zonoset_P0 = obj_reach.compute_zonoset(act_zono['c'], act_zono['g'])
        obj_visual.filled_polygon(zonoset_P0, 'green')
    for act_zono in X:
        zonoset_P0 = obj_reach.compute_zonoset(act_zono['c'], act_zono['g'])
        #obj_visual.filled_polygon(zonoset_P0, 'orange')
    traj=obj_reach.sampling_trajectory(np.matrix([[2],[2],[2], [2]]))
    obj_visual.show_traj(traj)
    obj_visual.show()