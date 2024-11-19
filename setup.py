from setuptools import setup
import os, re

long_description = """
EBDM (Error-Based-Deformation Method) is a B-spline fitting algorithm specifically developed to fit a template multipatch
geometry of the ventricle onto sparse data. It can be used for arbitrary multipatch geometries and pointcloud sparsity.
The resulting geometry is used as a basis for the CardIGA module (IGA-based cardiac model) to perform patient-pecific analyses.
"""
with open(os.path.join('ebdm', '__init__.py')) as f:
  version = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)

setup(name     = 'ebdm',
      version  = '2.0',
      author   = 'Robin Willems, Lex Verberne',
      packages = ['ebdm'],
      description      = 'Error-Based-Deformation Method for multipatch geometries',
      download_url     = 'https://github.com/CardiacIGA/multipatch-fitter/tree/main',
      long_description = long_description,
      zip_safe = False)