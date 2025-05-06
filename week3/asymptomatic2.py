import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.dates as mdates

# --- Data Loading & Preprocessing ---
# Read CSV file (replace with your actual filename)
try:
    df = pd.read_csv('../CovidIstErkrankungsbeginn.csv', parse_dates=['Meldedatum', 'Refdatum'])
except FileNotFoundError:
    raise FileNotFoundError("CSV file not found. Please ensure the file exists and the path is correct.")

# Handle missing dates and values
df = df.dropna(subset=['Meldedatum', 'Refdatum', 'AnzahlFall', 'IstErkrankungsbeginn'])

# Calculate detection delay (in days)
df['DetectionDelay'] = (df['Meldedatum'] - df['Refdatum']).dt.days

# --- Data Processing ---
# Process symptomatic/asymptomatic cases with proper date handling
def process_cases(df, symptomatic_flag):
    cases = (df[df['IstErkrankungsbeginn'] == symptomatic_flag]
             .groupby('Meldedatum')['AnzahlFall']
             .sum()
             .reset_index())
    cases = cases.set_index('Meldedatum')
    full_dates = pd.date_range(df['Meldedatum'].min(), df['Meldedatum'].max(), freq='D')
    return cases.reindex(full_dates, fill_value=0).sort_index()

symptomatic = process_cases(df, 1)
asymptomatic = process_cases(df, 0)

dates = symptomatic.index
days = len(dates)

# --- Model Parameters ---
germany_pop = 83.2e6  # 83.2 million
initial_exposed = max(1000, int(symptomatic.iloc[0] * 2))  # Heuristic for initial exposed

# --- SEAILRS Model (Extended with Asymptomatics) ---
def seailrs_model(params, pop=germany_pop):
    beta, sigma, alpha, gamma, delta, p = params  # p = fraction asymptomatic
    
    # Initialize compartments
    S = np.zeros(days)  # Susceptible
    E = np.zeros(days)  # Exposed
    I = np.zeros(days)  # Symptomatic infectious
    A = np.zeros(days)  # Asymptomatic infectious
    L = np.zeros(days)  # Reported cases (symptomatic)
    R = np.zeros(days)  # Recovered
    
    # Initial conditions (using first data point)
    I[0] = symptomatic.iloc[0]/pop
    A[0] = float(asymptomatic.iloc[0]/pop)
    E[0] = initial_exposed/pop
    R[0] = 0
    S[0] = 1 - sum([I[0], A[0], E[0], R[0]])

    for t in range(days-1):
        # Time-varying transmission (accounting for interventions)
       # current_beta = beta * 0.7 if t > 30 else beta
        
        # New infections (from both I and A compartments)
        new_infections = beta * S[t] * (I[t] + 0.5*A[t])  # Asymptomatics 50% as infectious
        
        # Model dynamics
        S[t+1] = S[t] - new_infections + delta * R[t]
        E[t+1] = E[t] + new_infections - sigma * E[t]
        I[t+1] = I[t] + (1-p)*sigma*E[t] - alpha*I[t]
        A[t+1] = A[t] + p*sigma*E[t] - alpha*A[t]
        L[t+1] = L[t] + alpha*I[t] - gamma*L[t]
        R[t+1] = R[t] + gamma*L[t] + alpha*A[t]
        
        # Numerical stability
        for comp in [S, E, I, A, L, R]:
            comp[t+1] = max(0, min(1, comp[t+1]))

    return {
        'S': S*pop,
        'E': E*pop,
        'I': I*pop,
        'A': A*pop,
        'L': L*pop,
        'R': R*pop
    }

# --- Optimization ---
def loss(params):
    compartments = seailrs_model(params)
    return np.sum((compartments['L'] - symptomatic['AnzahlFall'].values)**2)

bounds = [
    (0.5, 3.0),   # β
    (1/7, 1/3),   # σ (incubation period 3-7 days)
    (0.1, 0.5),   # α
    (0.05, 0.3),  # γ
    (0.0, 0.005), # δ
    (0.2, 0.6)    # p (asymptomatic fraction)
]

# Initial parameter guesses
initial_guess = [
    2.5,           # β
    1/5,           # σ (5 day incubation)
    0.2,           # α
    0.1,           # γ
    0.001,         # δ
    0.4            # p
]

result = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')
best_params = result.x

# --- Results ---
compartments = seailrs_model(best_params)

# --- Visualization ---
plt.figure(figsize=(14, 8))
plt.plot(dates, symptomatic['AnzahlFall']/1e6, color='blue' , label='Actual Symptomatic')
plt.plot(dates, compartments['L']/1e6,color='green' , label='Modeled Reported Cases')
plt.plot(dates, asymptomatic['AnzahlFall']/1e6, color='black', label='Actual Asymptomatic')
plt.plot(dates, compartments['A']/1e6, color='purple', label='Modeled Asymptomatic')

plt.title('Germany COVID-19 SEAILRS Model Fit\nIncluding Asymptomatic Cases', fontsize=14, pad=20)
plt.ylabel('Cases (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Compartment plot
plt.figure(figsize=(14, 8))
plt.stackplot(dates, 
             compartments['S']/1e6,
             compartments['E']/1e6,
             (compartments['I'] + compartments['A'])/1e6,
             compartments['L']/1e6,
             compartments['R']/1e6,
             labels=['Susceptible', 'Exposed', 'Infectious (I+A)', 'Reported (L)', 'Recovered'],
             colors=['blue', 'orange', 'green', 'red', 'purple'])

plt.title('Population Compartment Dynamics', fontsize=14, pad=20)
plt.ylabel('Population (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Parameter output
print(f"""\nOptimized Parameters:
• Transmission rate (β): {best_params[0]:.3f}
• Incubation rate (σ): {best_params[1]:.3f} (≈{1/best_params[1]:.1f} days)
• Infectious rate (α): {best_params[2]:.3f}
• Recovery rate (γ): {best_params[3]:.3f}
• Waning immunity (δ): {best_params[4]:.5f}
• Asymptomatic fraction (p): {best_params[5]:.2f}

Reproduction Numbers:
• R₀ (basic): {best_params[0]*(1 + 0.5*best_params[5])/best_params[2]:.2f}
""")

plt.show()