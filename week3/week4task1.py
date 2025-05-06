import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Load Data ---
df = pd.read_csv('../CovidIstErkrankungsbeginn.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])
#df = pd.read_csv('./covid file.csv')

pivoted = df.pivot_table(index='Meldedatum', columns='Bundesland', values='AnzahlFall', aggfunc='sum').fillna(0)
pivot_recovered = df.pivot_table(index='Meldedatum', values='AnzahlGenesen', aggfunc='sum').fillna(0)

dates = pivoted.index
states = pivoted.columns.tolist()
real_data_matrix = pivoted.values.T

# --- Estimate cumulative recoveries as R ---
real_Recovered = pivot_recovered.cumsum().values.flatten()

# --- Sum all states infected into one (Germany as a whole) ---
real_infected = np.sum(real_data_matrix, axis=0)

# --- Germany population ---
germany_total_population = 8320
days = len(dates)

# --- SEILRS model for entire Germany ---
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    # Initial conditions (1st real data point as infected)
    I[0] = 50
    #I[0] = real_infected[0]
    E[0] = 0
    L[0] = real_infected[0]
    R[0] = real_Recovered[0]
    S[0] = germany_total_population - I[0] - E[0] - L[0] - R[0]

    for k in range(days - 1):
        new_infections = beta * S[k] * I[k] / germany_total_population
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return S, E, I, L, R  # Return all compartments

# --- Loss function for optimization ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    _, _, _, L_sim, _ = seilrs_model(beta, sigma, alpha, gamma, delta)
    return np.sum((L_sim - real_infected) ** 2)

# --- Optimize parameters ---
initial_guess = [0.8, 0.4, 0.46, 0.05, 0.0175]
bounds = [(0.1, 1.5), (0.05, 1.2), (0.1, 1.0), (0.001, 0.3), (0, 0.03)]

result = minimize(loss, initial_guess, bounds=bounds)
best_params = result.x

print("\n✅ Best-fit parameters:")
print(f"β = {best_params[0]:.4f}")
print(f"σ = {best_params[1]:.4f}")
print(f"α = {best_params[2]:.4f}")
print(f"γ = {best_params[3]:.4f}")
print(f"δ = {best_params[4]:.4f}")

# --- Run model with best parameters ---
S, E, I, L, R = seilrs_model(*best_params)

# --- Plot all compartments ---
plt.figure(figsize=(14, 8))

plt.plot(dates, S, label='Susceptible (S)', color='blue')
plt.plot(dates, E, label='Exposed (E)', color='orange')
plt.plot(dates, I, label='Infectious (I)', color='green')
plt.plot(dates, L, label='Symptomatic (L)', color='red')
plt.plot(dates, R, label='Recovered (R)', color='purple')
plt.plot(dates, real_infected, label='Reported Infections (Actual)', color='black', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Number of People (in millions)')
plt.title('SEILRS Model Compartments vs. Actual COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()