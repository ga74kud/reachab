import matplotlib.pyplot as plt
import numpy as np
import reachab as rb
from scipy.optimize import linprog
from scipy import signal
from scipy.linalg import block_diag

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

a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
b = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
c = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
d = np.array([[0, 0], [0, 0]])
cont_sys=signal.StateSpace(a, b, c, d)
disc_sys=cont_sys.to_discrete(0.1)
rt=block_diag(disc_sys.A, disc_sys.B, disc_sys.A, disc_sys.B)
lin_solv=linprog(c=[-1., -1., 1, 1, 1, 1,-1., -1., 1, 1, 1, 1,],
    A_ub=[[1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [-1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
          ],
    b_ub=[1., 1., 1., 1., 4., -2., 4., -2.])
print(lin_solv)
None
import cdd
mat = cdd.Matrix([[2,-1,-1,0],[0,1,0,0],[0,0,1,0]], number_type='fraction')
mat.rep_type = cdd.RepType.INEQUALITY
poly = cdd.Polyhedron(mat)
print(poly)

mat = cdd.Matrix([[1, 1, -1], [1, 1, 1], [1, -1, 1], [1, -1, -1]], number_type='fraction')
mat.rep_type = cdd.RepType.GENERATOR
poly = cdd.Polyhedron(mat)
print(poly)
print(poly.get_inequalities())