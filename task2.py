import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider

def seirs_model(beta, sigma, gamma, delta, days=300):
    N = 10000
    S = np.zeros(days+1)
    E = np.zeros(days+1)
    I = np.zeros(days+1)
    R = np.zeros(days+1)

    # Initial conditions
    I[0] = 1
    E[0] = 0
    R[0] = 0
    S[0] = N - I[0] - E[0] - R[0]

    for k in range(days):
        S[k+1] = S[k] - (beta * S[k] * I[k]) / N + delta * R[k]
        E[k+1] = E[k] + (beta * S[k] * I[k]) / N - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - gamma * I[k]
        R[k+1] = R[k] + gamma * I[k] - delta * R[k]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible', color='blue')
    plt.plot(E, label='Exposed', color='orange')
    plt.plot(I, label='Infectious', color='red')
    plt.plot(R, label='Recovered', color='green')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SEIRS Model (Discrete & Interactive)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Interactive sliders
interact(
    seirs_model,
    beta=FloatSlider(value=0.4, min=0.1, max=1.0, step=0.05, description='β (Infection)'),
    sigma=FloatSlider(value=0.2, min=0.05, max=1.0, step=0.05, description='σ (Incubation)'),
    gamma=FloatSlider(value=0.1, min=0.05, max=1.0, step=0.05, description='γ (Recovery)'),
    delta=FloatSlider(value=0.01, min=0.0, max=0.2, step=0.01, description='δ (Loss of Immunity)')
)
