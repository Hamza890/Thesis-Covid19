import matplotlib.pyplot as plt
import numpy as np
#xpoints = np.array([1, 8])
#ypoints = np.array([3, 10])
#plt.plot(xpoints, ypoints)
#plt.show()


# Model parameters
B = 0.5      # Transmission rate (equivalent to beta)
gamma = 0.1  # Recovery rate
delta = 0    # Immunity loss rate (equivalent to xi)
N = 10000    # Total population 
days = 300   # Simulation duration

# Initial conditions
I0 = 1000    # Initial infected
R0 = 0       # Initial recovered
S0 = N - I0 - R0  # Initial susceptible

# Initialize arrays to store population values
S = np.zeros(days + 1)
I = np.zeros(days + 1)
R = np.zeros(days + 1)

# Set initial values
S[0] = S0
I[0] = I0
R[0] = R0

# Perform daily iterations
for k in range(days):
    # Current population values
    Sk = S[k]
    Ik = I[k]
    Rk = R[k]
    
    # Update equations
    S[k+1] = Sk - (B/N) * Ik * Sk + delta * Rk
    I[k+1] = Ik + (B/N) * Ik * Sk - gamma * Ik
    R[k+1] = Rk + gamma * Ik - delta * Rk
    
    # Optional: Ensure non-negative populations
    S[k+1] = max(S[k+1], 0)
    I[k+1] = max(I[k+1], 0)
    R[k+1] = max(R[k+1], 0)
    print(R[k+1])
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.title('Discrete-Time SIRS Model Simulation (300 Days)')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()