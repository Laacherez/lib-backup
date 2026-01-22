# Physical parameters for the run.py script. Simu units for now.

import numpy as np

# Geometry

a = 5e-5  # Colloid radius

hmin = 2.1 * a  # minimal colloid height, apex to wall distance
hmax = 3.0 * a  # maximal colloid height, apex to wall distance
hbins = 5  # how many of those heights to simulate
h = np.linspace(hmin, hmax, hbins)  # array

dmin = 2.1 * a  # minimal colloid-to-colloid distance, apex to apex distance
dmax = 10.0 * a  # maximal colloid-to-colloid distance, apex to apex distance
dbins = 5  # how many of those distances to simulate
d = np.linspace(dmin, dmax, dbins)  # array


periodic_length = 20*a
# Physics

rho_fluid = 1000.0  # Volumic mass of the fluid
rho_bead = 1050.0  # Volumic mass of the bead
eta = 1e-3  # viscosity
kbT = 4e-21  # thermal energy

# Simulation

dt = 0.000001


