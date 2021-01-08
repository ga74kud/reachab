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


class reachability(object):
    def __init__(self, **kwargs):
        self.zonotype= {'c': np.matrix([[2], [3]]), 'g': np.matrix([[2, 2], [3, 4]])}

if __name__ == '__main__':
    obj_reach = reachability()
    obj_visual = visualizer()
    obj_visual.show_point(obj_reach.zonotype['c'])
    zonoset=[[2, 3, 4, 3], [1, 4, 2, -2]]
    obj_visual.filled_polygon(zonoset)

    obj_visual.show()