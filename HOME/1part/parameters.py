# Physical parameters for the run.py script. Simu units for now.

import numpy as np

# Geometry

a = 5e-6  # Colloid radius

hmin = 2.1 * a  # minimal colloid height, apex to wall distance
hmax = 3.0 * a  # maximal colloid height, apex to wall distance
hbins = 5  # how many of those heights to simulate
h = np.linspace(hmin, hmax, hbins)  # array

periodic_length = 1e-3
# Physics

rho_fluid = 1000.0    # Volumic mass of the fluid
rho_bead = 1050.0   # Volumic mass of the bead
eta = 1e-3     # viscosity
kbT = 4e-21  # thermal energy
g = 9.81        # m/s^2


# Simulation

dt = 0.001


