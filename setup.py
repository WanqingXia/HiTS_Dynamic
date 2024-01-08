from setuptools import setup, find_packages  
setup(
    name='HiTS_Obs',
    packages=find_packages(),
    install_requires=["matplotlib", "scipy", "tianshou==0.3.1", "graph_rl==0.1.2", "dyn_rl_benchmarks"
                      , "pyglet==1.5.11", "mujoco-py<2.2,>=2.1"],
)