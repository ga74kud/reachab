![](/images/pexels-free-creative-stuff-1193743.jpg)


# Reachability Analysis
Reachability Analysis (RA) for hybrid systems:

- [x] RA with zonotypes

# Installation
```bash
pip install reachab
```

# Usage

After installation with ```bash pip install ``` and running the script:
```python
import reachab

def test_reachab():
    reachab.test_me()

if __name__ == '__main__':
    test_reachab()
```

... should produce:

![](/images/erg.png)


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




# Citation

```
TODO
```


Image source: https://www.pexels.com/photo/photo-of-multicolored-abstract-painting-1193743/