import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Load Data ---
df = pd.read_csv('./filtered_dataCovidIstErkrankungsbeginn1.csv')
df['Meldedatum(Reporting date)'] = pd.to_datetime(df['Meldedatum(Reporting date)'])

pivoted = df.pivot_table(
    index='Meldedatum(Reporting date)',
    columns='Bundesland',
    values='AnzahlFall(NumberCase)',
    aggfunc='sum'
).fillna(0)

dates = pivoted.index
real_data_matrix = pivoted.values.T  # shape: (states, days)

# --- Aggregate total daily infections in Germany ---
real_infected = np.sum(real_data_matrix, axis=0)
days = len(real_infected)
germany_total_population = 83_200_000

# --- Smooth real data ---
real_infected_smooth = pd.Series(real_infected).rolling(
    window=7, center=True
).mean().fillna(method='bfill').fillna(method='ffill').values

# --- SEILRS Model ---
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    # Seed compartments realistically
    I[0] = 200
    E[0] = 3000
    L[0] = 500
    R[0] = 0
    S[0] = germany_total_population - (E[0] + I[0] + L[0] + R[0])

    for k in range(days - 1):
        new_infections = beta * S[k] * I[k] / germany_total_population
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return L

# --- Loss Function ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    L_sim = seilrs_model(beta, sigma, alpha, gamma, delta)

    # Reporting lag: shift model to match reported data
    lag = 5
    L_shifted = np.pad(L_sim[lag:], (0, lag), mode='edge')

    # Log-scale loss (to handle wide range of values)
    return np.sum((np.log1p(L_shifted) - np.log1p(real_infected_smooth))**2)

# --- Optimize Parameters ---
initial_guess = [0.8, 0.4, 0.46, 0.05, 0.0175]
bounds = [(0.1, 1.5), (0.05, 1.2), (0.1, 1.0), (0.001, 0.3), (0.0, 0.03)]

result = minimize(loss, initial_guess, bounds=bounds)
best_params = result.x

print("\n✅ Best-fit parameters (constant):")
print(f"β = {best_params[0]:.4f}")
print(f"σ = {best_params[1]:.4f}")
print(f"α = {best_params[2]:.4f}")
print(f"γ = {best_params[3]:.4f}")
print(f"δ = {best_params[4]:.4f}")

# --- Simulate final model ---
best_L = seilrs_model(*best_params)
L_lagged = np.pad(best_L[5:], (0, 5), mode='edge')

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(dates, real_infected_smooth, label='Reported Infections (Smoothed)', color='black')
plt.plot(dates, L_lagged, label='Modeled Symptomatic (L)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of People')
plt.title('Improved SEILRS Fit to COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()