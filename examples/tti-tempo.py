# %%
# Demonstration of iTEBD within OQuPy
# Uses the original iTEBD code by Valentin Link from https://github.com/val-link/iTEBD-TEMPO.git.
# Minimally modified by Paul Eastham to take a bath correlation object from OQuPy.
# The resulting object can turned into a new proces tensor subclass, TTInvariantProcessTensor, which should function as all other PTs in OQuPy
# Todo: fully integrate the creation of iTEBD process tensors so we can skip calls to iTEBD_TEMPO_oqupy class.

# %%
import sys
sys.path.insert(0,'..')

import oqupy as oqupy
import numpy as np
import matplotlib.pyplot as plt

# %%
sigma_z = oqupy.operators.sigma("z")
sigma_x = oqupy.operators.sigma("x")
sigma_y = oqupy.operators.sigma("y")
omega_cutoff = 3.04 
alpha = 0.126
temperature = 0.1309
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(sigma_x/2.0, correlations)
parameters=oqupy.TempoParameters(dt=0.1,epsrel=1e-7,dkmax=100)
pt=oqupy.TTITempo(bath,start_time=0.0,parameters=parameters)
pt.compute()

# %%
detuning = lambda t: 0.0 * t
def gaussian_shape(t, area = 1.0, tau = 1.0, t_0 = 0.0):
    return area/(tau*np.sqrt(np.pi)) * np.exp(-(t-t_0)**2/(tau**2))

def hamiltonian_t(t):
    return detuning(t)/2.0 * sigma_z \
           + gaussian_shape(t, area = np.pi/2.0, tau = 0.5)/2.0 * sigma_y

system = oqupy.TimeDependentSystem(hamiltonian_t)

Rho_0=oqupy.operators.spin_dm('x-')

dynamics = oqupy.compute_dynamics(
       process_tensor=pt.get_process_tensor(),        
       system=system,
       initial_state=Rho_0,
       start_time=-5.0,
       num_steps=200)

t, s_x = dynamics.expectations(sigma_x, real=True)
_, s_y = dynamics.expectations(sigma_y, real=True)
_, s_z = dynamics.expectations(sigma_z, real=True)

plt.plot(t,s_x,label='sx')
plt.plot(t,s_y,label='sy')
plt.plot(t,s_z,label='sz')
plt.ylim((-1.0,1.0))
plt.legend(loc=4)

# %%



