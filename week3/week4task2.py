import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import timedelta

# --- Load Data ---
df = pd.read_csv('../CovidIstErkrankungsbeginn.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])

# Process data
pivoted = df.pivot_table(index='Meldedatum', columns='Bundesland', values='AnzahlFall', aggfunc='sum').fillna(0)
pivot_recovered = df.pivot_table(index='Meldedatum', values='AnzahlGenesen', aggfunc='sum').fillna(0)

dates = pivoted.index
states = pivoted.columns.tolist()
real_data_matrix = pivoted.values.T

# --- Germany population (in millions) ---
germany_total_population = 83.2  # 83.2 million
initial_infected = 460  # More realistic starting point

# --- Prepare real data ---
real_Recovered = pivot_recovered.cumsum().values.flatten() / germany_total_population
real_infected = np.sum(real_data_matrix, axis=0) / germany_total_population
days = len(dates)

# --- SEILRS model (normalized version) ---
def seilrs_model(beta, sigma, alpha, gamma, delta, dt=1.0):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    L = np.zeros(days)
    R = np.zeros(days)

    # Normalized initial conditions
    I[0] = initial_infected / germany_total_population
    E[0] = 0
    L[0] = real_infected[0]/ germany_total_population  # Start with first data point
    R[0] = real_Recovered[0]/ germany_total_population
    S[0] = 1.0 - I[0] - E[0] - L[0] - R[0]  # Total population normalized to 1

    for k in range(days - 1):
        
        # Reduce β after 30 days (simulating lockdowns)
        current_beta = beta * 0.7 if dates[k] > dates[0] + timedelta(days=30) else beta
        new_infections = current_beta * S[k] * I[k] * dt


        S[k+1] = max(0, S[k] - new_infections + delta * R[k] * dt)
        E[k+1] = max(0, E[k] + new_infections - sigma * E[k] * dt)
        I[k+1] = max(0, I[k] + sigma * E[k] * dt - alpha * I[k] * dt)
        L[k+1] = max(0, L[k] + alpha * I[k] * dt - gamma * L[k] * dt)
        R[k+1] = max(0, R[k] + gamma * L[k] * dt - delta * R[k] * dt)

        # Ensure total population remains ~1.0 (optional)
        total = S[k+1] + E[k+1] + I[k+1] + L[k+1] + R[k+1]
        if abs(total - 1.0) > 0.01:  # Allow small numerical errors
            S[k+1] /= total
            E[k+1] /= total
            I[k+1] /= total
            L[k+1] /= total
            R[k+1] /= total

    # Convert back to absolute numbers (in millions)
    return (S * germany_total_population,
            E * germany_total_population,
            I * germany_total_population,
            L * germany_total_population,
            R * germany_total_population)

# --- Loss function ---
def loss(params):
    beta, sigma, alpha, gamma, delta = params
    _, _, _, L_sim, _ = seilrs_model(beta, sigma, alpha, gamma, delta)
    errors = (L_sim - (real_infected * germany_total_population)) ** 2
    weights = np.linspace(3.0, 1.0, len(errors))  # Higher weight for early points
    return np.sum(errors * weights)

# --- Optimize parameters ---
initial_guess = [3.0, 0.5, 0.2, 0.05, 0.01]  # Adjusted initial guesses
bounds = [(0.01, 3.0),   # β
          (0.01, 0.7),    # σ
          (0.01, 0.5),    # α
          (0.001, 0.3),   # γ
          (0.0, 0.01)]     # δ

result = minimize(loss, initial_guess, bounds=bounds)
best_params = result.x

print("\n✅ Best-fit parameters:")
print(f"β (transmission rate) = {best_params[0]:.4f}")
print(f"σ (incubation rate) = {best_params[1]:.4f}")
print(f"α (infectious rate) = {best_params[2]:.4f}")
print(f"γ (recovery rate) = {best_params[3]:.4f}")
print(f"δ (waning immunity) = {best_params[4]:.4f}")

# --- Run model with best parameters ---
S, E, I, L, R = seilrs_model(*best_params)




#Plot log-scale to see exponential phase
plt.subplot(2, 1, 1)
plt.yscale('log')  # Add this line to check early growth rate

# Print early values
print("Early time points (Model vs Actual):")
for i in [0, 10, 20, 30]:
    print(f"{dates[i]}: Model={L[i]:.1f}, Actual={real_infected[i] * germany_total_population:.1f}")

    
# --- Plot results ---
plt.figure(figsize=(8, 7))

# Main plot (L vs real data)
plt.subplot(2, 1, 1)
plt.plot(dates, real_infected * germany_total_population, 
         label='Reported Cases (Actual)', color='black', linewidth=2)
plt.plot(dates, L, label='Modeled Symptomatic (L)', color='red', linestyle='--')
plt.ylabel('Cases (millions)')
plt.title('SEILRS Model vs Actual COVID-19 Cases in Germany')
plt.legend()
plt.grid(True)

# All compartments plot
plt.subplot(2, 1, 2)
plt.plot(dates, S, label='Susceptible (S)')
plt.plot(dates, E, label='Exposed (E)')
plt.plot(dates, I, label='Infectious (I)')
plt.plot(dates, L, label='Symptomatic (L)')
plt.plot(dates, R, label='Recovered (R)')
plt.ylabel('Population (millions)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Print final values ---
print("\nFinal compartment sizes (millions):")
print(f"S: {S[-1]:.2f}, E: {E[-1]:.2f}, I: {I[-1]:.2f}, L: {L[-1]:.2f}, R: {R[-1]:.2f}")
print(f"Total: {S[-1]+E[-1]+I[-1]+L[-1]+R[-1]:.2f} (should be ~{germany_total_population:.1f})")

# --- Reproduction number ---
R0 = best_params[0] / best_params[2]  # β/α
print(f"\nBasic reproduction number R₀ = {R0:.2f}")