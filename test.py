import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ---- Step 1: Load & Preprocess Data ----
df = pd.read_csv('./covid file.csv')

# Rename and parse date
df.rename(columns={"Meldedatum(Reporting date)": "Date", "AnzahlFall(NumberCase)": "Infected"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

# Aggregate across all states (Germany-wide total per day)
df_total = df.groupby("Date")["Infected"].sum().reset_index()
dates = df_total["Date"]
real_infected = df_total["Infected"].values
days = len(real_infected)

# ---- Step 2: Model Constants ----
N = 83_200_000  # Total population of Germany (2020)

# ---- Step 3: SEILRS Model Definition ----
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    I[0] = real_infected[0]
    S[0] = N - I[0]

    for k in range(days - 1):
        new_infections = (beta * S[k] * I[k]) / N
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return L

# ---- Step 4: Loss Function ----
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    L_sim = seilrs_model(beta, sigma, alpha, gamma, delta)
    return np.sum((L_sim - real_infected) ** 2)

# ---- Step 5: Optimize Parameters ----
initial_guess = [0.8, 0.4, 0.46, 0.05, 0.0175]
bounds = [(0.1, 1.5), (0.05, 1.2), (0.1, 1.0), (0.001, 0.3), (0, 0.03)]

result = minimize(loss, initial_guess, bounds=bounds)
best_params = result.x
print("Best-fit parameters:")
print(f"β = {best_params[0]:.4f}, σ = {best_params[1]:.4f}, α = {best_params[2]:.4f}, γ = {best_params[3]:.4f}, δ = {best_params[4]:.4f}")

# ---- Step 6: Plot Results ----
best_L = seilrs_model(*best_params)

plt.figure(figsize=(12, 6))
plt.plot(dates, real_infected, label='Reported Infections (Actual)', color='black')
plt.plot(dates, best_L, label='Modeled Symptomatic (L)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of People')
plt.title('Fitting SEILRS Model to COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
