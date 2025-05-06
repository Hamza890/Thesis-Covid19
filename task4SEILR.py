import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Germany total population (2020) ---
germany_total_population = 83_200_000

# --- Federal states population (approx. 2020) ---
german_states_population = {
    'Baden-Württemberg': 11_100_000,
    'Bayern': 13_100_000,
    'Berlin': 3_769_000,
    'Brandenburg': 2_521_000,
    'Bremen': 681_000,
    'Hamburg': 1_847_000,
    'Hessen': 6_265_000,
    'Mecklenburg-Vorpommern': 1_609_000,
    'Niedersachsen': 7_982_000,
    'Nordrhein-Westfalen': 17_932_000,
    'Rheinland-Pfalz': 4_093_000,
    'Saarland': 983_000,
    'Sachsen': 4_077_000,
    'Sachsen-Anhalt': 2_194_000,
    'Schleswig-Holstein': 2_903_000,
    'Thüringen': 2_143_000
}

# --- Scale state populations so total = 83,200,000 ---
real_total = sum(german_states_population.values())
scaling_factor = germany_total_population / real_total
scaled_population = {state: int(pop * scaling_factor) for state, pop in german_states_population.items()}


# --- Load Data ---
df = pd.read_csv('./filtered_dataCovidIstErkrankungsbeginn1.csv')
#df = pd.read_csv('./covid file.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])

pivoted = df.pivot_table(index='Meldedatum', values='AnzahlFall', aggfunc='sum').fillna(0)
pivot_recovered = df.pivot_table(index='Meldedatum', values='AnzahlGenesen', aggfunc='sum').fillna(0)

dates = pivoted.index
states = pivoted.columns.tolist()
real_data_matrix = pivoted.values.T

# --- Sum all states into one (Germany as a whole) ---
real_infected = np.sum(real_data_matrix, axis=0)

# --- Estimate cumulative recoveries as R ---
real_R = pivot_recovered.cumsum().values.flatten()

# --- Germany population ---
germany_total_population = 83_200_000
days = len(dates)
# --- SEILRS model function ---
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros(( days))
    E = np.zeros((days))
    I = np.zeros(( days))
    L = np.zeros(( days))
    R = np.zeros(( days))

     # Initial conditions
    I[0] = max(real_infected[0], 10)
    E[0] = 0
    L[0] = 0
    R[0] = real_R[0]
    S[0] = germany_total_population - (E[0] + I[0] + L[0] + R[0])

    for k in range(days - 1):
        new_infections = beta * S[k] * I[k] / germany_total_population
        S[k+1] = S[k] - new_infections + delta * R[k]
        E[k+1] = E[k] + new_infections - sigma * E[k]
        I[k+1] = I[k] + sigma * E[k] - alpha * I[k]
        L[k+1] = L[k] + alpha * I[k] - gamma * L[k]
        R[k+1] = R[k] + gamma * L[k] - delta * R[k]

    return S, E, I, L, R

# --- Loss function ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    _, _, _, L_model, _ = seilrs_model(beta, sigma, alpha, gamma, delta)
    min_len = min(len(L_model), len(real_infected))
    return np.sum((L_model[:min_len] - real_infected[:min_len]) ** 2)

# --- Run model with example parameters ---
beta = 0.35    # infection rate
sigma = 0.2    # incubation rate (5 days avg)
alpha = 0.5    # 2-day avg from I → symptomatic
gamma = 0.1    # 10-day recovery
delta = 0.005  # moderate immunity loss (every 200 days)

# --- Parameter Optimization ---
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

# --- Simulate with best-fit parameters ---
S, E, I, L, R = seilrs_model(*best_params)

# --- Plot ---
plt.figure(figsize=(14, 7))
plt.plot(dates, real_infected, label='Reported Infections (Actual)', color='black', linewidth=2)
plt.plot(dates, L, label='Modeled Symptomatic (L)', linestyle='--', color='red')
plt.plot(dates, I, label='Infectious (I)', linestyle='--', color='orange')
plt.plot(dates, E, label='Exposed (E)', linestyle='--', color='purple')
plt.plot(dates, R, label='Recovered (R)', linestyle='--', color='green')
plt.plot(dates, S, label='Susceptible (S)', linestyle='--', color='blue')

plt.xlabel('Date')
plt.ylabel('Number of People')
plt.title('Fitting SEILRS Model to COVID-19 Infections in Germany')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
