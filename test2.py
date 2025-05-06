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

# --- Load COVID infection dataset ---
df = pd.read_csv('./covid file.csv')
#df.rename(columns={"Meldedatum(Reporting date)": "Date",'Bundesland':"States", "AnzahlFall(NumberCase)": "Infected"}, inplace=True)
df["Meldedatum(Reporting date)"] = pd.to_datetime(df["Meldedatum(Reporting date)"])

# --- Pivot infection data ---
pivoted = df.pivot_table(index='Meldedatum(Reporting date)', columns='Bundesland', values='AnzahlFall(NumberCase)', aggfunc='sum').fillna(0)
real_data_matrix = pivoted.values.T
dates = pivoted.index
states = pivoted.columns.tolist()
real_infected = pivoted.values.tolist()

#real_infected = ["Infected"].values
days = len(real_infected)


# --- Population vector for states ---
N_states = np.array([scaled_population[state] for state in states])
num_states = len(states)
days = len(dates)

# --- SEILRS model function ---
def seilrs_model(beta, sigma, alpha, gamma, delta):
    S = np.zeros((num_states, days))
    E = np.zeros((num_states, days))
    I = np.zeros((num_states, days))
    L = np.zeros((num_states, days))
    R = np.zeros((num_states, days))

    I[:,0] = real_infected[ 0]
    E[:, 0] = 0
    L[:, 0] = 0
    R[:, 0] = 0
    S[ :,0] = N_states - I[:, 0] - E[:, 0] - R[ :,0] - L[ :,0]

    for k in range(days - 1):
        new_infections = (beta * S[:, k] * I[:, k]) / N_states
        S[:, k+1] = S[:, k] - new_infections + delta * R[:, k]
        E[:, k+1] = E[:, k] + new_infections - sigma * E[ :,k]
        I[:, k+1] = I[ :,k] + sigma * E[ :,k] - alpha * I[:, k]
        L[:, k+1] = L[:, k] + alpha * I[:, k] - gamma * L[:, k]
        R[:, k+1] = R[:, k] + gamma * L[:, k] - delta * R[:, k]

    return S, E, I, L, R

# ---- Step 4: Loss Function ----
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    L_sim ,*_= seilrs_model(beta, sigma, alpha, gamma, delta)
    return np.sum((L_sim - real_infected) ** 2)

# ---- Step 5: Optimize Parameters ----
initial_guess = [0.8, 0.4, 0.46, 0.05, 0.0175]
bounds = [(0.1, 1.5), (0.05, 1.2), (0.1, 1.0), (0.001, 0.3), (0, 0.03)]

result =minimize(loss, initial_guess, bounds=bounds)
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