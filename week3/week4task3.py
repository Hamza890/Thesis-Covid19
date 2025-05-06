import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

# --- Data Loading ---
df = pd.read_csv('../filtered_dataCovidIstErkrankungsbeginn1.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])
df['Refdatum'] = pd.to_datetime(df['Refdatum'])

# Process case data by cumulative sum
cases = df.groupby('Meldedatum')['AnzahlFall'].sum().cumsum()
recovered = df.groupby('Meldedatum')['AnzahlGenesen'].sum().cumsum()
dates = cases.index
real_I = cases.values
real_R = recovered.values

# --- Parameters ---
germany_pop = 83.2e6
initial_infected = 1000
days = len(real_I)  # Critical: Match model length to data length

# --- SEILRS Model ---
def seilrs_model(params, pop=germany_pop):
    beta, sigma, alpha, gamma, delta = params
    
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)
    
    # Initial conditions
    I[0] = initial_infected/pop
    #I[0] = real_I[0]/pop
    E[0] = 0
    L[0] = real_I[0]/pop
    R[0] = real_R[0]/pop
    S[0] = 1 - I[0] - E[0] - L[0] - R[0]

    for t in range(days-1):
        #current_beta = beta * 0.7 if t > 30 else beta
        new_infections = beta * S[t] * I[t]
        
        S[t+1] = max(0, S[t] - new_infections + delta * R[t])
        E[t+1] = max(0, E[t] + new_infections - sigma * E[t])
        I[t+1] = max(0, I[t] + sigma * E[t] - alpha * I[t])
        L[t+1] = max(0, L[t] + alpha * I[t] - gamma * L[t])
        R[t+1] = max(0, R[t] + gamma * L[t] - delta * R[t])

    return (S*pop, E*pop, I*pop, L*pop, R*pop)

# --- Optimization ---
def loss(params):
    _, _, I_, _, _ = seilrs_model(params)
    return np.sum((I_ - real_I)**2)

bounds = [
    (0.5, 3.0), (1/10, 1/3), 
    (0.1, 0.5), (0.05, 0.3),
    (0.0, 0.005)
]

result = minimize(loss, [2.0, 0.14, 0.2, 0.1, 0.001], bounds=bounds)
best_params = result.x

# --- Results ---
S, E, I, L, R = seilrs_model(best_params)

plt.figure(figsize=(8, 8))
#plt.plot(dates, S/1e6, label='Susceptible (S)', color='blue')
#plt.plot(dates, E /1e6, label='Exposed (E)', color='orange')
plt.plot(dates, I/1e6, label='Infectious (I)', color='green')
plt.plot(dates, real_I /1e6, label='Reported Infections (Actual)', color='black', linestyle='--')
#plt.plot(dates, L/1e6, label='Symptomatic (L)', color='red')
#plt.plot(dates, R/1e6, label='Recovered (R)', color='purple')
plt.legend()
plt.ylabel('Cases (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)


# Plotting

plt.figure(figsize=(12, 6))
plt.plot(dates, real_I/1e6, 'ko', markersize=4, label='Actual Cases')
plt.plot(dates, I/1e6, 'r-', linewidth=2, label='Model Fit')
plt.title('Germany COVID-19 SEILRS Model Fit', fontsize=14)
plt.ylabel('Cases (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)
#/1e6
plt.figure(figsize=(14, 8))

plt.plot(dates, S/1e6, label='Susceptible (S)', color='blue')
plt.plot(dates, E/1e6, label='Exposed (E)', color='orange')
plt.plot(dates, I/1e6, label='Infectious (I)', color='green')
plt.plot(dates, L/1e6/1e6, label='Symptomatic (L)', color='red')
plt.plot(dates, R/1e6, label='Recovered (R)', color='purple')
plt.plot(dates, real_I/1e6, label='Reported Infections (Actual)', color='black', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Number of People (in millions)')
plt.title('SEILRS Model Compartments vs. Actual COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.show()




print(f"""Optimized Parameters:
β: {best_params[0]:.3f}
σ: {best_params[1]:.3f} (1/{1/best_params[1]:.1f} days)
α: {best_params[2]:.3f}
γ: {best_params[3]:.3f}
δ: {best_params[4]:.5f}

R₀: {best_params[0]/best_params[2]:.2f}""")