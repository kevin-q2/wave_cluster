from setuptools import find_packages, setup

setup(
    name='wave_cluster',
    author='Kevin Quinn',
    author_email='quinnk@bu.edu',
    packages=find_packages(include=['wave_cluster']),
    install_requires=['numpy', 'pandas', 'scipy', 'networkx', 'matplotlib', 'scikit-learn', 'multiprocessing'],
    version='0.1',
    description='cluster waves of a disease related time-series'
)