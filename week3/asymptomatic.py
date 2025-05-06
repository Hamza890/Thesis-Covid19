import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.dates as mdates


# --- Data Loading & Preprocessing ---
df = pd.read_csv('../CovidIstErkrankungsbeginn.csv')
df['Meldedatum'] = pd.to_datetime(df['Meldedatum'], dayfirst=True)#Reporting Date
df['Refdatum'] = pd.to_datetime(df['Refdatum'], dayfirst=True)#(symptom onset) is closer to the actual infection time


# Calculate detection delay
df['DetectionDelay'] = (df['Meldedatum'] - df['Refdatum']).dt.days

# Get all unique dates first
all_dates = df['Refdatum'].sort_values().unique()
dates = pd.DatetimeIndex(all_dates)  # Use this as the common time axis

# Create cumulative sums with reindexing to ensure same length
symptomatic = (
    df[df['IstErkrankungsbeginn'] == 1]
    .groupby('Refdatum')['AnzahlFall']
    .sum()
    .reindex(dates, fill_value=0)
    .cumsum()
)
asymptomatic = (
    df[df['IstErkrankungsbeginn'] == 0]
    .groupby('Refdatum')['AnzahlFall']
    .sum()
    .reindex(dates, fill_value=0)
    .cumsum()
)
# Process case data by cumulative sum with same dates
cases = (
    df.groupby('Refdatum')['AnzahlFall']
    .sum()
    .reindex(dates, fill_value=0)
    .cumsum()
)
recovered = (
    df.groupby('Refdatum')['AnzahlGenesen']
    .sum()
    .reindex(dates, fill_value=0)
    .cumsum()
)

# Now all series will have the same length
days = len(dates)
real_L = cases.values
real_R = recovered.values

# --- Data Loading & Preprocessing ---
#df = pd.read_csv('../CovidIstErkrankungsbeginn.csv')
#df['Meldedatum'] = pd.to_datetime(df['Meldedatum'])
#df['Refdatum'] = pd.to_datetime(df['Refdatum'])

# Calculate detection delay
#df['DetectionDelay'] = (df['Meldedatum'] - df['Refdatum']).dt.days

# Split symptomatic/asymptomatic cases
#symptomatic = df[df['IstErkrankungsbeginn'] == 1].groupby('Meldedatum')['AnzahlFall'].sum().cumsum()
#asymptomatic = df[df['IstErkrankungsbeginn'] == 0].groupby('Meldedatum')['AnzahlFall'].sum().cumsum()
#dates = symptomatic.index
#days = len(dates)

# Process case data by cumulative sum
#cases = df.groupby('Meldedatum')['AnzahlFall'].sum().cumsum()
#recovered = df.groupby('Meldedatum')['AnzahlGenesen'].sum().cumsum()
#dates = cases.index
#real_L = cases.values
#real_R = recovered.values

# --- Parameters ---
germany_pop = 83.2e6
initial_Infected = 1000
p_asymptomatic = 0.4  # Initial estimate, will be optimized

# --- SEAILRS Model (Extended with Asymptomatics) ---
def seailrs_model(params, pop=germany_pop):
    beta, sigma, alpha, gamma, delta, p = params  # p = fraction asymptomatic
    
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)  # Symptomatic infectious
    A = np.zeros(days)  # Asymptomatic infectious
    L = np.zeros(days)  # Symptomatic reported
    R = np.zeros(days)
    
    # Initial conditions (using first data point)
    I[0] = symptomatic.iloc[0]/pop
    A[0] = asymptomatic.iloc[0]/pop
    E[0] = 0
    R[0] = real_R[0]/pop
    L[0] = real_L[0]/pop
    S[0] = 1 - I[0] - A[0] - E[0] - R[0] - L[0]

    for t in range(days-1):
        
        # Time-varying transmission (accounting for interventions)
        #current_beta = beta * 0.7 if t > 30 else beta

    
        # New infections (from both I and A compartments)
        new_infections = beta * S[t] * (I[t] + 0.5*A[t])  # Asymptomatics 50% as infectious
        
        # Model dynamics
        S[t+1] = S[t] - new_infections + delta * R[t]
        E[t+1] = E[t] + new_infections - sigma * E[t]
        I[t+1] = I[t] + (1-p)*sigma*E[t] - alpha*I[t]  # Symptomatic
        A[t+1] = A[t] + p*sigma*E[t] - alpha*A[t]      # Asymptomatic
        L[t+1] = L[t] + alpha*I[t] - gamma*L[t]        # Only symptomatic become reported
        R[t+1] = R[t] + gamma*L[t] + alpha*A[t]        # Both types recover
        
        # Numerical stability
        #for comp in [S, E, I, A, L, R]:
            #comp[t+1] = max(0, min(1, comp[t+1]))
            


    return (S*pop, E*pop, I*pop, A*pop, L*pop, R*pop)

# --- Optimization ---
def loss(params):
    _, _, I_, _, _, _ = seailrs_model(params)
    return np.sum((I_ - symptomatic.values)**2)

bounds = [
    (0.5, 3.0),   # β
    (1/10, 1/3),  # σ
    (0.1, 0.5),   # α
    (0.05, 0.3),  # γ
    (0.0, 0.005), # δ
    (0.1, 0.6)    # p (asymptomatic fraction)
]

result = minimize(loss, [2.0, 0.14, 0.2, 0.1, 0.001, 0.4], bounds=bounds)
best_params = result.x

# --- Results ---
S, E, I, A, L, R = seailrs_model(best_params)

# --- Visualization ---
plt.figure(figsize=(14,7))
plt.plot(dates, symptomatic/1e6, label='Actual Symptomatic' ,color='black', linestyle='--')
plt.plot(dates, I/1e6, label='Modeled Reported Cases')
plt.plot(dates, asymptomatic/1e6, label='Actual Asymptomatic', color='Grey', linestyle='--')
plt.plot(dates, A/1e6,  label='Modeled Asymptomatic')

plt.title('Germany COVID-19 SEAILRS Model Fit', fontsize=14)
plt.ylabel('Cases (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.gcf().autofmt_xdate()
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Compartment plot
# --- Plot all compartments ---
plt.figure(figsize=(8, 8))

#plt.plot(dates, S/1e6, label='Susceptible (S)', color='blue')
#plt.plot(dates, E/1e6, label='Exposed (E)', color='orange')
plt.plot(dates, (I+A)/1e6, label='Infectious (I+A)', color='green')
#plt.plot(dates, L/1e6, label='Symptomatic/Reported (L)', color='red')
#plt.plot(dates, R/1e6, label='Recovered (R)', color='purple')
plt.plot(dates, real_L/1e6, label='Reported Infections (Actual)', color='black', linestyle='--')

plt.title('Population Compartments Over Time', fontsize=14)
plt.ylabel('Population (millions)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.gcf().autofmt_xdate()
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print(f"""Optimized Parameters:
β (Transmission): {best_params[0]:.3f}
σ (Incubation): {best_params[1]:.3f} (≈{1/best_params[1]:.1f} days)
α (Infectious): {best_params[2]:.3f}
γ (Recovery): {best_params[3]:.3f}
δ (Waning immunity): {best_params[4]:.5f}
p (Asymptomatic fraction): {best_params[5]:.2f}

Reproduction Number R₀: {best_params[0]*(1+0.5*best_params[5])/best_params[2]:.2f}
""")
plt.show()