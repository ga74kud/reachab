![GitHub Logo](/images/pexels-free-creative-stuff-1193743.jpg)
Image source: ![https://www.pexels.com/photo/photo-of-multicolored-abstract-painting-1193743/](url)


# Reachability Analysis
Reachability Analysis (RA) for hybrid systems:

- [x] RA with zonotypes

#Installation
```bash
pip install reachab
```

#Usage

```python
Omega_0 = {'c': np.matrix([[0],
                               [0],
                               [10],
                               [0]
                               ]),
               'g': np.matrix([[1, -1, 1, .2, .2],
                               [1, 1, .3, .2, .5],
                               [0, 0, 0, .4, .3],
                               [0, 0, 0, .2, .4]
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
```

#Citation

```
Hartmann, M.; Reachability Analysis in Python, 2021
```