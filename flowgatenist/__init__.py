"""
`flowgatenist`: Automated gating of flow cytometry data.

based, in part, on FlowCal package version 1.1.4
https://taborlab.github.io/FlowCal/index.html

for Python version 3

"""

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '1.0'

from . import io

from . import gaussian_mixture
from . import metadata
