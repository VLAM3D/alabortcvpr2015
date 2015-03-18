from setuptools import setup, find_packages
import numpy as np

include_dirs = [np.get_include()]

requirements = ['menpo>=0.4.4',
                'menpofit>=0.1.0']

setup(name='alabortcvpr2015',
      version='0.0.1',
      description='Repository containing the code of the paper: Unifying '
                  'Holistic and Parts-Based Deformable Models',
      author='Joan Alabort-i-Medina',
      author_email='joan.alabort@gmail.com',
      include_dirs=include_dirs,
      packages=find_packages(),
      install_requires=requirements)
