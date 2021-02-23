# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------
from __init__ import *
import numpy as np
import argparse
import logging
def run_it(params):
    #test_me()
    for wlt in range(0, 1):
        Omega_0 = {'c': np.matrix([[wlt*80],
                               [0],
                               [10],
                               [3]
                               ]),
               'g': np.matrix([[1, -1],
                               [1, 1],
                               [0, 0],
                               [0, 0]
                               ])
               }
        U = {'c': np.matrix([[0],
                         [0],
                         [0],
                         [0],
                         ]),
         'g': np.matrix([[1, 0],
                         [0, 1],
                         [0, 0],
                         [0, 3]
                         ])
         }
        # zonoset=reach(Omega_0, U, params)
        R, X, obj_reach, zonoset=reach_zonotype_without_box(Omega_0, U, **{"time_horizon": 2.2, "steps": 4, "visualization": "y", "face_color": "green"})
        all_inside_points=points_inside_hull(zonoset)
        plot_all_inside_points(all_inside_points)
        logging.info("Numbers in num_list are: {}".format(' '.join(map(str, zonoset))))
    show_all()



def main():
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
    if(params['debug']=='y'):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    run_it(params)
if __name__ == '__main__':
    main()

