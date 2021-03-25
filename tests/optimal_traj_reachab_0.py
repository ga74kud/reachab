import matplotlib.pyplot as plt
import numpy as np
import reachab as rb
from scipy.optimize import linprog



Omega_0 = {'c': np.matrix([[0],
                               [0],
                               [0],
                               [0]
                               ]),
               'g': np.matrix([[10, 0],
                               [0, 10],
                               [0, 0],
                               [0, 0]
                               ])
               }
U = {'c': np.matrix([[50],
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
R, X, obj_reach, zonoset=rb.reach_zonotype_without_box(Omega_0, U, **{"time_horizon": 2.2, "steps": 4, "visualization": "y", "face_color": "green"})

plt.show()


lin_solv=linprog(c=[1., 1., 1., 1.],
    A_ub=[[1., 0.0, 0.0, 0.0], [-1., 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1., 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
    b_ub=[1., 1., 1., 1., 4., -2., 4., -2.],
    bounds=(None, None))
print(lin_solv)
None