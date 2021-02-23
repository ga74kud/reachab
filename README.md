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

[comment]: <> (```python)

[comment]: <> (    parser = argparse.ArgumentParser&#40;&#41;)

[comment]: <> (    parser.add_argument&#40;'--box_function', '-box', type=str, help='choices: without_box, with_box',)

[comment]: <> (                        default='without_box', required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--visualization', '-vis', type=str, help='y, n',)

[comment]: <> (                        default='y', required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--time_horizon', '-T', type=float, help='value like: T=2.2', default=2.2, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--steps', '-N', type=int, help='value like N=4', default=6, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--debug', '-deb', type=str, help='&#40;y,n&#41;', default='n', required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--window_x', '-wix', type=int, help='windowsize in x-direction for savgol_filter', default=101, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--window_y', '-wiy', type=int, help='windowsize in y-direction for savgol_filter', default=101, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--poly_x', '-pox', type=int, help='polygon order in x-direction for savgol_filter', default=2, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--poly_y', '-poy', type=int, help='polygon order in y-direction for savgol_filter', default=2, required=False&#41;)

[comment]: <> (    parser.add_argument&#40;'--program', '-pro', type=str, help='a&#41; only_reachability', default='only_reachability', required=False&#41;)

[comment]: <> (    args = parser.parse_args&#40;&#41;)

[comment]: <> (    params = vars&#40;args&#41;)

[comment]: <> (    params['PROJECT_ROOT']=definitions.get_project_root&#40;&#41;)

[comment]: <> (    if &#40;params['debug'] == 'y'&#41;:)

[comment]: <> (        logging.basicConfig&#40;format='%&#40;levelname&#41;s:%&#40;message&#41;s', level=logging.DEBUG&#41;)

[comment]: <> (    only_reachability&#40;params&#41;)

[comment]: <> (```)

```python
import reachab as rb
import numpy as np
import matplotlib.pyplot as plt
Omega_0 = {'c': np.matrix([[80],
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
R, X, obj_reach, zonoset=rb.reach_zonotype_without_box(Omega_0, U, **{"time_horizon": 2.2, "steps": 4, "visualization": "y", "face_color": "green"})
all_inside_points=rb.points_inside_hull(zonoset)
rb.plot_all_inside_points(all_inside_points)
plt.grid()
plt.show()
```

... should produce:

![](/images/reachab.png)


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