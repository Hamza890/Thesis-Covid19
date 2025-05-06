import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Load Data ---
df = pd.read_csv('../filtered_dataCovidIstErkrankungsbeginn1.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])

# --- Aggregate data for all of Germany ---
pivot_cases = df.pivot_table(index='Meldedatum', values='AnzahlFall', aggfunc='sum').fillna(0)
pivot_recovered = df.pivot_table(index='Meldedatum', values='AnzahlGenesen', aggfunc='sum').fillna(0)

# --- Smooth daily new infections ---
real_L = pivot_cases.values.flatten()
real_L_smooth = pd.Series(real_L).rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

# --- Estimate cumulative recoveries as R ---
real_R = pivot_recovered.cumsum().values.flatten()
real_R = pd.Series(real_R).fillna(method='bfill').fillna(method='ffill').values

# --- Setup date index and parameters ---
dates = pivot_cases.index
days = len(dates)
population = 83_200_000

# --- SEILRS model with fixed initial conditions ---
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    # Seed values (guesses based on early data)
    L[0] = real_L_smooth[0]
    E[0] = 0
    I[0] = 1
    R[0] = real_R
    S[0] = population - (E[0] + I[0] + L[0] + R[0])

    for k in range(days - 1):
        new_infections = beta * S[k] * I[k] / population
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return L, R

# --- Loss function ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    L_sim, _ = seilrs_model(beta, sigma, alpha, gamma, delta)
    
    lag = 5
    L_lagged = np.pad(L_sim[lag:], (0, lag), mode='edge')
    return np.mean((L_lagged - real_L_smooth) ** 2)
    #return np.sum((np.log1p(L_lagged) - np.log1p(real_L_smooth)) ** 2)

# --- Optimize parameters ---
initial_guess = [0.8, 0.4, 0.46, 0.05, 0.0175]
bounds = [(0.1, 1.5), (0.05, 1.2), (0.1, 1.0), (0.001, 0.3), (0, 0.03)]

result = minimize(loss, initial_guess, bounds=bounds)
best_params = result.x

# --- Output Best Parameters ---
print("\n✅ Best-fit parameters:")
print(f"β = {best_params[0]:.4f}")
print(f"σ = {best_params[1]:.4f}")
print(f"α = {best_params[2]:.4f}")
print(f"γ = {best_params[3]:.4f}")
print(f"δ = {best_params[4]:.4f}")

# --- Run model with best parameters ---
best_L, best_R = seilrs_model(*best_params)

# --- Plot L compartment (Symptomatic) vs Real Data ---
plt.figure(figsize=(12, 6))
plt.plot(dates, real_L_smooth, label='Reported Infections (Smoothed)', color='black')
plt.plot(dates, best_L, label='Modeled Symptomatic (L)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of People')
plt.title('Improved SEILRS Fit to COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
