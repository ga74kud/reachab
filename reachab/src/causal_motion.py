# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Causal Inference and Motion Planning
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------

from reachab.util.visualizer import *
from reachab.util.util_functions import *
import numpy as np


class causal_motion(object):
    def __init__(self, **kwargs):
        self.x = None
        self.prob=None
    def set_initial_x(self, x):
        self.x=x
    def last_pos(self):
        return self.x[-1]
    def set_initial_prob(self, prob):
        self.prob=prob

    def move(self):
        pos=self.x[-1]
        samp=self.sample()
        erg=(pos[0]+samp[0], pos[1]+samp[1])
        self.x.append(erg)
    def sample(self):
        mean = self.prob["mean"]
        cov = self.prob["covariance"]  # diagonal covariance
        dif_x, dif_y = np.random.multivariate_normal(mean, cov, 1).T
        return (np.float(dif_x), np.float(dif_y))

if __name__ == '__main__':
    obj_causal_A=causal_motion()
    obj_visual=visualizer()

    obj_causal_A.set_initial_x([(-10, 0)])
    obj_causal_A.set_initial_prob({"mean": [2, 0], "covariance": [[1, 0], [0, 1]]})
    obj_causal_B = causal_motion()
    obj_causal_B.set_initial_x([(10, 0)])
    obj_causal_B.set_initial_prob({"mean": [-2, 0], "covariance": [[1, 0], [0, 1]]})
    for wlt in range(0, 10):
        distance=distance_2_points(obj_causal_A.last_pos(), obj_causal_B.last_pos())
        if(distance<3):
            obj_causal_A.set_initial_prob({"mean": [0, 2], "covariance": [[1, 0], [0, 1]]})
            obj_visual.line_between(obj_causal_A.last_pos(), obj_causal_B.last_pos(), "red")

        else:
            obj_visual.line_between(obj_causal_A.last_pos(), obj_causal_B.last_pos(), "green")

        obj_causal_A.move()
        obj_causal_B.move()
    print(obj_causal_A.x)
    print(obj_causal_B.x)
    obj_visual.show_traj(obj_causal_A.x)
    obj_visual.show_traj(obj_causal_B.x)
    obj_visual.show()