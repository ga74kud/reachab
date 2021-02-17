import argparse
import numpy as np
import reachab as rb
import matplotlib.pyplot as plt

##################
### Parameters ###
##################
parser = argparse.ArgumentParser()
parser.add_argument('--box_function', '-box', type=str, help='choices: without_box, with_box',
                    default='without_box', required=False)
parser.add_argument('--visualization', '-vis', type=str, help='(y,n)',
                    default='n', required=False)
parser.add_argument('--time_horizon', '-T', type=float, help='value like: T=2.2', required=True)
parser.add_argument('--steps', '-N', type=int, help='value like N=4', required=True)
parser.add_argument('--debug', '-deb', type=str, help='(y,n)', default='n', required=False)
parser.add_argument('--face_color', '-facol', type=str, help='name: orange, green or values', default='cyan', required=False)
parser.add_argument('--nx', '-nx', type=int, help='nx for meshgrid (integer)', default=6,
                    required=False)
parser.add_argument('--ny', '-ny', type=int, help='ny for meshgrid (integer)', default=6,
                    required=False)
args = parser.parse_args()
params = vars(args)

####################################
### Compute synthetic trajectory ###
####################################
Ts=params['time_horizon']/(params['steps']+1)
x=[]
y=[]
x.append(0)
y.append(0)
vx=[]
vy=[]
vx.append(4)
vy.append(3)
ax=[]
ay=[]
ax.append(.1)
ay.append(1.2)
for wlt in range(0, 10):
    new_x, new_y = rb.compute_next_position(vx[-1], vy[-1], x[-1], y[-1], Ts)
    new_vx, new_vy = rb.compute_next_velocity(vx[-1], vy[-1], ax[-1], ay[-1], Ts)
    x.append(new_x)
    y.append(new_y)
    vx.append(new_vx)
    vy.append(new_vy)
    total_distance=rb.compute_cumulative_distance(x, y)
plt.plot(x, y)
plt.grid()
plt.show()
# Omega_0 = {'c': np.matrix([[80],
#                                [0],
#                                [10],
#                                [3]
#                                ]),
#                'g': np.matrix([[1, -1],
#                                [1, 1],
#                                [0, 0],
#                                [0, 0]
#                                ])
#                }
# U = {'c': np.matrix([[0],
#                  [0],
#                  [0],
#                  [0],
#                  ]),
#  'g': np.matrix([[1, 0, 1],
#                  [1, 1, 0],
#                  [0, 0, 0],
#                  [0, 0, 0]
#                  ])
#  }
# points=rb.get_points_zonotype(U)
# zonoset=rb.reach(Omega_0, U, params)
#
# rb.show_all()