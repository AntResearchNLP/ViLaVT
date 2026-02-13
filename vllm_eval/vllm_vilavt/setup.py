# from setuptools import setup

# setup(name='vllm_vilavt',
#       version='0.1',
#       packages=['vllm_vilavt'],
#       entry_points={
#           'vllm.general_plugins':
#           ["register_vilavt_model = vllm_vilavt:register"]
#       })

# setup.py
from setuptools import setup, find_packages

setup(
    name="vllm_vilavt",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins":
        ["register_vilavt_model = vllm_vilavt:register"]
    }
)
