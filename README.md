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