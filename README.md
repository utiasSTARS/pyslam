# pyslam
Non-linear least-squares SLAM in Python using scipy and numpy. Modelled after Google's Ceres solver.

Dependencies:
* numpy (for most things)
* scipy (for sparse linear algebra)
* numba (for vectorization speedups)
* [liegroups](https://github.com/utiasSTARS/liegroups)

### Installation
To install, `cd` into the repository directory (the one with `setup.py`) and run:
```
pip install -e .
```
The `-e` flag tells pip to install the package in-place, which lets you make changes to the code without having to reinstall every time.

### Testing
Ensure you have `pytest` installed on your system, or install it using `conda install pytest` or `pip install pytest`. Then run `pytest` in the repository directory.
