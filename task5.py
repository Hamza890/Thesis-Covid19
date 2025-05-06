import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Load Data ---
df = pd.read_csv('./filtered_dataCovidIstErkrankungsbeginn1.csv')

df['Meldedatum(Reporting date)'] = pd.to_datetime(df['Meldedatum(Reporting date)'])

pivoted = df.pivot_table(index='Meldedatum(Reporting date)', columns='Bundesland', values='AnzahlFall(NumberCase)', aggfunc='sum').fillna(0)
dates = pivoted.index
states = pivoted.columns.tolist()
real_data_matrix = pivoted.values.T  # shape: (states, days)

# --- Sum all states into one (Germany as a whole) ---
real_infected = np.sum(real_data_matrix, axis=0)

# --- Smooth the data with a 7-day moving average ---
real_infected_smoothed = pd.Series(real_infected).rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

# --- Germany population ---
germany_total_population = 83_200_000
days = len(dates)

# --- SEILRS model for entire Germany with initial seeding ---
def seilrs_model(beta, sigma, alpha, gamma, delta, E0=3000, L0=200):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    # Initial conditions
    I[0] = real_infected[0]
    E[0] = E0
    L[0] = L0
    R[0] = 0
    S[0] = germany_total_population - I[0] - E[0] - L[0] - R[0]

    for k in range(days - 1):
        new_infections = beta * S[k] * I[k] / germany_total_population
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return L  # Return only L for fitting

# --- Loss function for optimization ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    L_sim = seilrs_model(beta, sigma, alpha, gamma, delta, E0=500, L0=200)

    # Apply lag to model output to account for symptom/report delay
    lag = 7
    if lag > 0:
        L_sim = np.pad(L_sim[lag:], (0, lag), mode='edge')

    return np.sum((L_sim - real_infected_smoothed) ** 2)

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

# --- Plot best-fit model vs smoothed real data ---
best_L = seilrs_model(*best_params, E0=500, L0=200)

# Apply the same lag for plotting
lag = 7
if lag > 0:
    best_L = np.pad(best_L[lag:], (0, lag), mode='edge')

plt.figure(figsize=(12, 6))
plt.plot(dates, real_infected_smoothed, label='Reported Infections (Smoothed)', color='black')
plt.plot(dates, best_L, label='Modeled Symptomatic (L)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of People')
plt.title('Improved SEILRS Fit to COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
