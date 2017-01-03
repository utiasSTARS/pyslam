from setuptools import setup

setup(
    name='pyslam',
    version='0.0.0',
    description='Non-linear least-squares SLAM in Python using scipy and numpy.',
    author='Lee Clement',
    author_email='lee.clement@robotics.utias.utoronto.ca',
    license='MIT',
    packages=['pyslam'],
    install_requires=['numpy', 'scipy', 'liegroups']
)
