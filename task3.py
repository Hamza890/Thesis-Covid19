import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load state-wise data (we’ll assume columns: State, Date, Infected)
df = pd.read_csv('./covid file.csv')
print (df)
states = df['Bundesland'].unique() # states
days = df['Refdatum'].nunique() #Dates

#states population approximately 2020
state_population_dict = {
    'Baden-Württemberg': 11100000,
    'Bayern': 13100000,
    'Berlin': 3769000,
    'Brandenburg': 2520000,
    'Bremen': 681000,
    'Hamburg': 1850000,
    'Hessen': 6280000,
    'Mecklenburg-Vorpommern': 1600000,
    'Niedersachsen': 7990000,
    'Nordrhein-Westfalen': 17900000,
    'Rheinland-Pfalz': 4090000,
    'Saarland': 986000,
    'Sachsen': 4080000,
    'Sachsen-Anhalt': 2190000,
    'Schleswig-Holstein': 2910000,
    'Thüringen': 2130000
}


pivoted = df.pivot_table(index='Meldedatum(Reporting date)', columns='Bundesland', values='AnzahlFall(NumberCase)', aggfunc='sum').fillna(0)
real_data_matrix = pivoted.values.T  # shape: (num_states, days)
dates = pivoted.index
states = pivoted.columns.tolist()

# --- Get population per state ---
N_states = np.array([state_population_dict[state] for state in states])
num_states = len(states)
days = len(dates)

# --- SEIRS vectorized model ---
def seirs_vector_model(beta, sigma, gamma, delta):
    S = np.zeros((num_states, days))
    E = np.zeros((num_states, days))
    I = np.zeros((num_states, days))
    R = np.zeros((num_states, days))

    # Initial conditions
    I[:, 0] = real_data_matrix[:, 0]
    E[:, 0] = 0
    R[:, 0] = 0
    S[:, 0] = N_states - I[:, 0] - E[:, 0] - R[:, 0]

    for k in range(days - 1):
        infection = (beta * S[:, k] * I[:, k]) / N_states
        S[:, k+1] = S[:, k] - infection + delta * R[:, k]
        E[:, k+1] = E[:, k] + infection - sigma * E[:, k]
        I[:, k+1] = I[:, k] + sigma * E[:, k] - gamma * I[:, k]
        R[:, k+1] = R[:, k] + gamma * I[:, k] - delta * R[:, k]

    return S, E, I, R

# --- Run the model with default parameters ---
beta, sigma, gamma, delta = 0.3, 0.2, 0.1, 0.01
S, E, I, R = seirs_vector_model(beta, sigma, gamma, delta)

# --- Plot: Simulated vs. Actual Infections for 4 sample states ---
plt.figure(figsize=(14, 6))
for idx, state in enumerate(states[:4]):  # Show only first 4 for clarity
    plt.plot(dates, I[idx], label=f"{state} (Simulated)")
    plt.plot(dates, real_data_matrix[idx], '--', label=f"{state} (Real)")

plt.xlabel("Date")
plt.ylabel("Infected")
plt.title("SEIRS Model vs. Real Infections (Sample States)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
