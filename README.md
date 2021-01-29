![](/images/pexels-free-creative-stuff-1193743.jpg)


# Reachability Analysis
Reachability Analysis (RA) for hybrid systems:

- [x] RA with zonotypes

# Installation
```bash
pip install reachab
```

# Usage

After installation with ```bash pip install reachab```, you could test the installation with ```reachab.test_me``` 
or running the script:
```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--box_function', '-box', type=str, help='choices: without_box, with_box',
                        default='without_box', required=False)
    parser.add_argument('--visualization', '-vis', type=str, help='y, n',
                        default='y', required=False)
    parser.add_argument('--time_horizon', '-T', type=float, help='value like: T=2.2', default=2.2, required=False)
    parser.add_argument('--steps', '-N', type=int, help='value like N=4', default=6, required=False)
    parser.add_argument('--debug', '-deb', type=str, help='(y,n)', default='n', required=False)
    parser.add_argument('--window_x', '-wix', type=int, help='windowsize in x-direction for savgol_filter', default=101, required=False)
    parser.add_argument('--window_y', '-wiy', type=int, help='windowsize in y-direction for savgol_filter', default=101, required=False)
    parser.add_argument('--poly_x', '-pox', type=int, help='polygon order in x-direction for savgol_filter', default=2, required=False)
    parser.add_argument('--poly_y', '-poy', type=int, help='polygon order in y-direction for savgol_filter', default=2, required=False)
    parser.add_argument('--program', '-pro', type=str, help='a) only_reachability', default='only_reachability', required=False)
    args = parser.parse_args()
    params = vars(args)
    params['PROJECT_ROOT']=definitions.get_project_root()
    if (params['debug'] == 'y'):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    only_reachability(params)
```

```python
    def only_reachability(params):
    Omega_0 = {'c': np.matrix([[0],
                               [0],
                               [10],
                               [3]
                               ]),
               'g': np.matrix([[1, -1, .5, .2],
                               [1, 1, -.8, .3],
                               [0, 0, .2, .3],
                               [0, 0, .1, .5]
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
    zonoset=reachab.reach(Omega_0, U, params)
    show_all()
```

... should produce:

![](/images/reachability.png)


# Citation

Please cite following document if you use this python package:
```
@inproceedings{hartmann2019optimal,
  title={Optimal motion planning with reachable sets of vulnerable road users},
  author={Hartmann, Michael and Watzenig, Daniel},
  booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)},
  pages={891--898},
  year={2019},
  organization={IEEE}
}
```


Image source: https://www.pexels.com/photo/photo-of-multicolored-abstract-painting-1193743/