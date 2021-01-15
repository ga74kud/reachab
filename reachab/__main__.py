from __init__ import *

import numpy as np
import argparse
import logging
def run_it(params):
    Omega_0 = {'c': np.matrix([[0],
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
         'g': np.matrix([[1, 0, 1],
                         [1, 1, 0],
                         [0, 0, 0],
                         [0, 0, 0]
                         ])
         }
    zonoset=reach(Omega_0, U, params)
    logging.info("Numbers in num_list are: {}".format(' '.join(map(str, zonoset))))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--box_function', '-box', type=str, help='choices: without_box, with_box',
                        default='without_box', required=False)
    parser.add_argument('--visualization', '-vis', type=str, help='y, n',
                        default='n', required=False)
    args = parser.parse_args()
    params = vars(args)
    run_it(params)
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    main()

