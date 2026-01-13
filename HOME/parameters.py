# Physical parameters for the run.py script, in SI.

import numpy as np

# Geometry

a = 10e-6 # Colloid radius

hmin = 1e-6
hmax = 20e-6
hbins = 10
h = np.linspace(hmin, hmax, hbins)

dmin =1e-6
dmax = 20e-6
dbins = 10
d = np.linspace(dmin, dmax, dbins)