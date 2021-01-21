# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

from reachab.util.visualizer import *
import numpy as np
import scipy.spatial
from scipy import signal
import logging

"""
    Class for reachability analysis
"""
class reachability(object):
    def __init__(self, **kwargs):
        self.params={'T': kwargs.get('T',2.2),
                     'N': kwargs.get('N',4),
                     'gamma': 0.01 #threshold for control input constraint (inf-norm)
                     }
        self.obj_visual = visualizer()
        self.params['r']=self.params['T']/(self.params['N']+1)
        self.sys={'A': None, 'B': None, 'C': None, 'D': None}
        self.init_fcn()

    """
       Initial function to get system dynamics
    """
    def init_fcn(self):
        A = np.matrix([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
        B = np.matrix([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1]])
        C = np.eye(4)
        D = np.zeros((4, 2))
        self.system_dynamics(A, B, C, D)

    """
        Initial step to start logging
    """
    def start_logging(self):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    """
        System dynamics
    """
    def system_dynamics(self, A, B, C, D):
        self.discrete_sys=signal.StateSpace(A, B, C, D, dt=self.params['r'])
        self.sys['A']=np.array(self.discrete_sys.A)
        self.sys['B'] = np.array(self.discrete_sys.B)
        self.sys['C'] = np.array(self.discrete_sys.C)
        self.sys['D'] = np.array(self.discrete_sys.D)

        #see Otto Föllinger, "Regelungstechnik" and Thesis of Matthias Althoff:
        self.Phi=np.eye(4)+self.params['r']*self.sys['A']+1/(2)*(self.sys['A']*self.params['r'])**2+1/(6)*(self.sys['A']*self.params['r'])**3+1/(24)*(self.sys['A']*self.params['r'])**4

    """
        Mulitplication with the center
    """
    def multiplication_on_center(self, mat):
        return mat*np.matrix(self.zonotype['c'])

    """
        Multiplication with a generator list
    """
    def multiplication_on_generator(self, mat, list):
        return mat*np.matrix(list)

    """
        Convert to Matrix
    """
    def convert_to_matrix(self, list):
        erg=np.hstack((list[0], list[1]))
        for i in range(2, len(list)):
            erg=np.hstack((erg, list[i]))
        return erg

    """
        Mulitplication on a zonotype
    """
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

    """
        Get unique vectors
    """
    def get_unique_vectors(self, vec):
        unique_vec = np.squeeze(np.unique(vec, axis=0))
        return unique_vec

    """
        Get edge points of zonotype with convex hull operation
    """
    def get_points_of_zonotype(self, zonotype):
        c=zonotype['c']
        g=zonotype['g']
        x_vec = [c]
        for wlt in range(0, np.size(g,1)):
            a=g[:, wlt]
            x_pos = [i+a for i in x_vec]
            x_neg = [i-a for i in x_vec]
            x_vec=x_pos+x_neg
        unique_vec=self.get_unique_vectors(x_vec)
        try:
            points = self.compute_convex_hull(unique_vec[:, 0], unique_vec[:, 1])
        except:
            None
        return [points[:, 0], points[:, 1]]

    """
        Compute the convex hull
    """
    def compute_convex_hull(self, x, y):
        v=np.transpose(np.vstack([x, y]))
        hull = scipy.spatial.ConvexHull(v)
        return hull.points[hull.vertices]

    """
        Minkowski sum with two zonotypes
    """
    def minkowski_zonotypes(self, ZA, ZB):
        Z={'c': None, 'g': None}
        Z['c']=ZA['c']+ZB['c']
        a=ZA['g']
        b=ZB['g']
        new_g=np.hstack((a, b))
        Z['g'] = new_g
        return Z

    # def square_zonotype(self, radius):
    #     Z={'c': np.array([[0],
    #                      [0],
    #                      [0],
    #                      [0],
    #                     [0]
    #                     ]),
    #        'g': None
    #        }
    #     b=[
    #            np.array([[radius], [0], [0], [0], [0]]),
    #            np.array([[0], [radius], [0], [0], [0]]),
    #            np.array([[0], [0], [radius], [0], [0]]),
    #            np.array([[0], [0], [0], [radius], [0]]),
    #            np.array([[0], [0], [0], [0], [radius]]),
    #        ]
    #     Z['g']=self.get_unique_vectors(b)
    #     return Z

    """
        Get Box Hull
    """
    def get_box_hull(self, Omega):
        Z = {'c': None, 'g': None}
        r=self.get_points_of_zonotype(Omega)
        std=(((np.max(r[0])-np.min(r[0]))/2, (np.max(r[1])-np.min(r[1]))/2))
        q=np.diag(std)
        Z['c']=Omega['c']
        Z['g'] = np.matrix(np.vstack((q,np.zeros((2,2)))))
        return Z

    """
        Approximation of Reachability analysis:
        based on algorithm 1 from: Girard, A.; "Efficient Computation of Reachable Sets of Linear Time-Invariant Systems 
        with Inputs"
    """
    def approximate_reachable_set_without_box(self, Omega_0, U):
        all_R = []
        all_X = []
        '''
        algorithm 1 from: Girard, A.; "Efficient Computation of Reachable Sets of Linear Time-Invariant Systems with 
        Inputs"
        '''
        all_R.append(Omega_0)
        all_X.append(Omega_0)
        #self.params['N']
        logging.info("number of generators Omega_0: " + str(np.size(Omega_0['g'], 1)))
        logging.info("number of generators U: " + str(np.size(U['g'], 1)))
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
                   'g': np.matrix([[0],
                                   [0],
                                   [0],
                                   [0]
                                   ])
                   }
        S_i=S_0
        logging.info("number of generators S_0: " + str(np.size(S_0['g'], 1)))
        # 4. step
        for i in range(0, self.params['N'] - 1):
            logging.info("cycle i: "+str(i))
            # 5. step
            X_i = self.multiplication_on_zonotype(self.Phi, X_i)
            all_X.append(X_i)
            logging.info("number of generators X_i: "+str(np.size(X_i['g'],1)))
            # 6. step
            S_i=self.minkowski_zonotypes(S_i, V_i)
            logging.info("number of generators S_i: "+str(np.size(S_i['g'],1)))
            # 7. step
            V_i = self.multiplication_on_zonotype(self.Phi, V_i)
            logging.info("number of generators V_i: "+str(np.size(V_i['g'],1)))
            # 8. step
            Omega_i=self.minkowski_zonotypes(X_i, S_i)
            logging.info("number of generators Omega_i: "+str(np.size(Omega_i['g'],1)))
            all_R.append(Omega_i)
        return all_R, all_X

    """
            Approximation of Reachability analysis:
            based on algorithm 2 from: Girard, A.; "Efficient Computation of Reachable Sets of Linear Time-Invariant 
            Systems with Inputs"
    """
    def approximate_reachable_set_with_box(self, Omega_0, U):
        all_R = []
        all_X = []
        '''
        algorithm 2 from: Girard, A.; "Efficient Computation of Reachable Sets of Linear Time-Invariant Systems with 
        Inputs"
        '''
        all_R.append(Omega_0)
        all_X.append(Omega_0)

        #self.params['N']
        logging.info("number of generators Omega_0: " + str(np.size(Omega_0['g'], 1)))
        logging.info("number of generators U: " + str(np.size(U['g'], 1)))
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
                   'g': np.matrix([[0],
                                   [0],
                                   [0],
                                   [0]
                                   ])
                   }
        S_i=S_0
        logging.info("number of generators S_0: " + str(np.size(S_0['g'], 1)))
        # 4. step
        for i in range(0, self.params['N'] - 1):
            logging.info("cycle i: "+str(i))
            # 5. step
            X_i = self.multiplication_on_zonotype(self.Phi, X_i)
            all_X.append(X_i)
            logging.info("number of generators X_i: "+str(np.size(X_i['g'],1)))
            # 6. step
            V_i_box = self.get_box_hull(V_i)
            S_i=self.minkowski_zonotypes(S_i, V_i_box)
            logging.info("number of generators S_i: "+str(np.size(S_i['g'],1)))
            # 7. step
            V_i = self.multiplication_on_zonotype(self.Phi, V_i)
            logging.info("number of generators V_i: "+str(np.size(V_i['g'],1)))
            # 8. step
            Omega_i=self.minkowski_zonotypes(X_i, S_i)
            Omega_i=self.get_box_hull(Omega_i)
            logging.info("number of generators Omega_i: "+str(np.size(Omega_i['g'],1)))
            all_R.append(Omega_i)
        return all_R, all_X

    """
        Approximation of Reachability analysis:
        based on algorithm from: Girard, A.; "Reachability of Uncertain Linear Systems
        Using Zonotopes"
    """
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

    """
        Center trajectory
    """
    def center_trajectory(self, R):
        erg=[]
        for i in range(0, len(R)):
            x = np.float(R[i]['c'][0])
            y = np.float(R[i]['c'][1])
            erg.append((x,y))
        return erg

    """
        Test function
    """
    def test_function(self):
        Omega_0 = {'c': np.matrix([[0],
                                   [0],
                                   [10],
                                   [3]
                                   ]),
                   'g': np.matrix([[1, -1, 1, .2, .2],
                                   [1, 1, .3, .2, .5],
                                   [0, 0, 0, .4, .3],
                                   [0, 0, 0, .2, .4]
                                   ])
                   }
        U = {'c': np.matrix([[0],
                             [0],
                             [0],
                             [0],
                             ]),
             'g': np.matrix([[1, 0, 1],
                             [1, 1, 0],
                             [0, 0, 0],
                             [0, 0, 0]
                             ])
             }
        program = ['without_box', 'with_box']
        program_select = 0
        if (program[0] == program[program_select]):
            R, X = self.approximate_reachable_set_without_box(Omega_0, U)
        elif (program[1] == program[program_select]):
            R, X = self.approximate_reachable_set_with_box(Omega_0, U)
        for act_zono in R:
            zonoset_P0 = self.get_points_of_zonotype(act_zono)
            self.obj_visual.filled_polygon(zonoset_P0, 'green', .2)
        for act_zono in X:
            zonoset_P0 = self.get_points_of_zonotype(act_zono)
            self.obj_visual.filled_polygon(zonoset_P0, 'orange')
        traj = self.center_trajectory(R)
        self.obj_visual.show_traj(traj)
        self.obj_visual.show()

if __name__ == '__main__':
    obj_reach = reachability(**{'T':3, 'N':5})
    obj_reach.start_logging()
    obj_reach.test_function()
