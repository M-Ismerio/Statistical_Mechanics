# Statistical Mechanics 2023.1
# Author: Marcelo Ismerio Oliveira, Bruno Honorato Scorzelli
# This code provides a numerical solution to the stochastic differential equation
# dT/dt = -U'(T,t) + gη(T)
# where U(T) = 16*U_0 * (T-T_1)*(T-T_1) * (T-T_2)*(T-T_2) / (T_1-T_2)**(-4) + epsilon*T*cos(w*t)

from numpy import *
from matplotlib import pyplot as plt
plt.style.use('seaborn')
font = {    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
        'size'   : 13}
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'stix'

# Potential
def U(T, t, parameters):
  U_0, T_1, T_2, epsilon, w, g = parameters
  return 16*U_0 * (T-T_1)*(T-T_1) * (T-T_2)*(T-T_2) / (T_1-T_2)**(4) + epsilon*T*cos(w*t)

# Force due to the potential and extra force
def U_prime(T, t, parameters):
  U_0, T_1, T_2, epsilon, w, g = parameters
  return 32*U_0 * (T-T_1)*(T-T_2) * (2*T-T_1-T_2) / (T_1-T_2)**(4) + epsilon*cos(w*t-pi/4) # a phase has been added for style. It is not relevant for the discussion.


# Gaussian noise of mean 0 and standard deviation 1
def eta(dt): 
  return random.normal(0,1) 

def solve_langevin(parameters, tmax=400, PLOT=True):
  
  # Define time step and range
  t0 = 0
  dt = 0.001
  time_range = arange(t0, tmax, dt)

  # Save arrays for plotting
  save_frequency = tmax/4
  save_times = []
  save_T = []

  T = parameters[1] # initial value
  g = parameters[-1]
  for step, t in enumerate(time_range):

    D1 = -U_prime(T, t, parameters)
    D2 = g*g
    T = T + D1*dt + sqrt(D2*dt)*eta(t)
    if step%save_frequency == 0:
        save_T.append(T)
        save_times.append(t)
  
  # Plotting (optional)
  if PLOT == True:
    T_2 = parameters[2]
    plt.style.use('seaborn')
    font = {    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
        'size'   : 22}
    plt.rc('font', **font)
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(save_times, save_T)
    ax.set_xticks(arange(0, pi/w*17, pi/w), arange(0, 17, 1), fontsize=16)
    ax.set_xlabel(r'$t$ [$10^5$ years]', fontsize=16)
    ax.set_yticks(arange(-2*T_2, 3*T_2, T_2), [r'$-2T_0$', r'$-T_0$', '0', r'$T_0$', r'$2T_0$'], fontsize=16)
    ax.set_ylabel(r'$T$', fontsize=16)
    plt.tight_layout()
    plt.savefig('EstMec_1_StochasticResonance.png', dpi=600)
    plt.show()

  return save_times, save_T

###########################################################################################
# Figure 0: Potential
parameters = [100, -10, 10, 5, 0.25/2, 7]
U_0, T_1, T_2, epsilon, w, g = parameters
T = linspace(-15, 15, 300)
plt.style.use('seaborn')
fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
ax[0].plot(T, U(T, 0, parameters))
ax[1].plot(T, U(T, pi/2/w, parameters))
ax[2].plot(T, U(T, pi/w, parameters))
ax[0].set_yticks(array([0, U_0]), [r'$0$', r'$U_0$'], fontsize=13)
ax[2].set_xlabel(r'$T$', fontsize=13)
ax[1].set_yticks(array([0, U_0]), [r'$0$', r'$U_0$'], fontsize=13)
ax[2].set_yticks(array([0, U_0]), [r'$0$', r'$U_0$'], fontsize=13)
ax[0].set_title(r'$U(T, t)$', fontsize=13)
ax[2].set_xticks(array([T_1, 0, T_2]), [r'$T_1$', r'$0$', r'$T_2$'], fontsize=13)
ax[0].text(0.5, 0.9, r'$t = 0$', transform=ax[0].transAxes, fontsize=13)
ax[1].text(0.5, 0.9, r'$t = \pi/2\omega$', transform=ax[1].transAxes, fontsize=13)
ax[2].text(0.5, 0.9, r'$t = \pi/\omega$', transform=ax[2].transAxes, fontsize=13)
plt.tight_layout()
plt.savefig('EstMec_0_Potential.png', dpi=600)
plt.show()

###########################################################################################
# Figure 1: Stochastic Ressonance
x, y = solve_langevin([100, -10, 10, 16, 0.25/2, 7], PLOT=True)

###########################################################################################
# Figure 2: PDF of the temperature.
# Plot Histogram for Prob Distribution
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(7, 5))
plt.yticks(fontsize=13)
plt.hist(y,ec="white", alpha=0.9, bins=30, density=True)
ax.set_xticks(arange(-15, 20, 5), [r'$-3T_0/2$', r'$-T_0$', r'$-T_0/2$', r'$0$', r'$T_0/2$', r'$T_0$', r'$3T_0/2$'], fontsize=13)
ax.set_xlabel(r'$T$', fontsize=13)
ax.set_ylabel(r'Probability Distribution', fontsize=13)
plt.tight_layout()
plt.savefig('EstMec_2_PDF.png', dpi=600)
plt.show()

###########################################################################################
# Figure 3: Comparison between different values of noise and amplitude. Everything else is fixed. 
fig, ax = plt.subplots(3, 1, figsize = (10,6), sharex=True)
T_2 = 10
# Low noise
ax1 = ax[0]
x, y = solve_langevin([100, -10, 10, 10, 0.25/2, 1], PLOT=False)
ax1.plot(x, y)
ax1.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)
ax1.set_ylabel(r'Low $g$', fontsize=16)

# High noise, high force strength 
ax2 = ax[1]
x, y = solve_langevin([100, -10, 10, 10, 0.25/2, 7], PLOT=False)
ax2.plot(x, y)
ax2.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)

# Low force strength
ax3 = ax[2]
x, y = solve_langevin([100, -10, 10, 1, 0.25/2, 7], PLOT=False)
ax3.plot(x, y)
ax3.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)
w = 0.25/2
ax3.set_xticks(arange(0, pi/w*17, pi/w), arange(0, 17, 1), fontsize=16)
ax3.set_xlabel(r'$t$ [$10^5$ years]', fontsize=16)
ax3.set_ylabel(r'Low $\varepsilon$', fontsize=16)

plt.savefig('EstMec_3_ComparisonAmplitude.png', dpi=600)
plt.show()

###########################################################################################
# Figure 4: Comparison between different values of frequency. Everything else is fixed. 
fig, ax = plt.subplots(3, 1, figsize = (10,6), sharex=True)
T_2 = 10
# High Frequency
ax1 = ax[0]
w = 1
x, y = solve_langevin([100, -10, 10, 17, w, 7], PLOT=False)
ax1.plot(x, y)
ax1.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)
ax1.set_ylabel(r'High $\omega$', fontsize=16)

# Kramer Frequency
ax2 = ax[1]
U_0 = 100
g = 7
w_k = sqrt(2)*U_0/10*exp(-2*U_0/g/g)/2
x, y = solve_langevin([100, -10, 10, 17, w_k, 7], PLOT=False)
ax2.plot(x, y)
ax2.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)

# Low frequency
ax3 = ax[2]
w = 0.05/5
x, y = solve_langevin([100, -10, 10, 17, w, 7], PLOT=False)
ax3.plot(x, y)
ax3.set_yticks(arange(-T_2, 2*T_2, T_2), [r'$-T_0$', '0', r'$T_0$'], fontsize=16)
ax3.set_xticks(arange(0, pi/w_k*16, pi/w_k), arange(0, 16, 1), fontsize=16)
ax3.set_xlabel(r'$t$ [$10^5$ years]', fontsize=16)
ax3.set_ylabel(r'Low $\omega$', fontsize=16)

plt.savefig('EstMec_4_ComparisonOmega.png', dpi=600)
plt.show()
