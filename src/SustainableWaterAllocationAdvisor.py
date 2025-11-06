import pandas as pd 
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
df = pd.read_csv('dataset/cleaned_global_water_consumption.csv')

# -----------------------------
# STEP 2: Ask for country input and filter dataset
# -----------------------------
while True:
    country_name = input("Enter country name: ").strip()
    country_df = df[df['Country'].str.lower() == country_name.lower()]

    if country_df.empty:
        print(f"No data found for '{country_name}'. Please try again.")
    else:
        print(f"Using dataset for {country_name} ({len(country_df)} records)")
        break
# -----------------------------
# STEP 3: Compute variable ranges
# -----------------------------
agri_min, agri_max = country_df['Agricultural Water Use (%)'].min(), country_df['Agricultural Water Use (%)'].max()
ind_min, ind_max   = country_df['Industrial Water Use (%)'].min(), country_df['Industrial Water Use (%)'].max()
rain_min, rain_max = country_df['Rainfall Impact (Annual Precipitation in mm)'].min(), country_df['Rainfall Impact (Annual Precipitation in mm)'].max()

print(f"\nAgriculture range: {agri_min:.2f} – {agri_max:.2f}")
print(f"Industry range:    {ind_min:.2f} – {ind_max:.2f}")
print(f"Rainfall range:    {rain_min:.2f} – {rain_max:.2f}")

# -----------------------------
# STEP 4: Define fuzzy variables
# -----------------------------
agriculture = ctrl.Antecedent(np.arange(agri_min, agri_max + 1, 1), 'agriculture')
industry    = ctrl.Antecedent(np.arange(ind_min, ind_max + 1, 1), 'industry')
rainfall    = ctrl.Antecedent(np.arange(rain_min, rain_max + 1, 1), 'rainfall')
scarcity    = ctrl.Consequent(np.arange(0, 101, 1), 'scarcity')

# Membership functions
agriculture['low']    = fuzz.trimf(agriculture.universe, [agri_min, agri_min, (agri_min + agri_max) / 2])
agriculture['medium'] = fuzz.trimf(agriculture.universe, [agri_min, (agri_min + agri_max) / 2, agri_max])
agriculture['high']   = fuzz.trimf(agriculture.universe, [(agri_min + agri_max) / 2, agri_max, agri_max])

industry['low']    = fuzz.trimf(industry.universe, [ind_min, ind_min, (ind_min + ind_max) / 2])
industry['medium'] = fuzz.trimf(industry.universe, [ind_min, (ind_min + ind_max) / 2, ind_max])
industry['high']   = fuzz.trimf(industry.universe, [(ind_min + ind_max) / 2, ind_max, ind_max])

rainfall['low']    = fuzz.trimf(rainfall.universe, [rain_min, rain_min, (rain_min + rain_max) / 2])
rainfall['medium'] = fuzz.trimf(rainfall.universe, [rain_min, (rain_min + rain_max) / 2, rain_max])
rainfall['high']   = fuzz.trimf(rainfall.universe, [(rain_min + rain_max) / 2, rain_max, rain_max])

scarcity['low']    = fuzz.trimf(scarcity.universe, [0, 0, 40])
scarcity['medium'] = fuzz.trimf(scarcity.universe, [20, 50, 80])
scarcity['high']   = fuzz.trimf(scarcity.universe, [60, 100, 100])

# -----------------------------
# STEP 5: Define fuzzy rules (LIST METHOD, 27 rules)
# -----------------------------
rules = [
    # Rainfall = LOW
    ctrl.Rule(rainfall['low'] & agriculture['low']    & industry['low'],    scarcity['medium']),
    ctrl.Rule(rainfall['low'] & agriculture['low']    & industry['medium'], scarcity['medium']),
    ctrl.Rule(rainfall['low'] & agriculture['low']    & industry['high'],   scarcity['high']),
    ctrl.Rule(rainfall['low'] & agriculture['medium'] & industry['low'],    scarcity['medium']),
    ctrl.Rule(rainfall['low'] & agriculture['medium'] & industry['medium'], scarcity['high']),
    ctrl.Rule(rainfall['low'] & agriculture['medium'] & industry['high'],   scarcity['high']),
    ctrl.Rule(rainfall['low'] & agriculture['high']   & industry['low'],    scarcity['high']),
    ctrl.Rule(rainfall['low'] & agriculture['high']   & industry['medium'], scarcity['high']),
    ctrl.Rule(rainfall['low'] & agriculture['high']   & industry['high'],   scarcity['high']),

    # Rainfall = MEDIUM
    ctrl.Rule(rainfall['medium'] & agriculture['low']    & industry['low'],    scarcity['low']),
    ctrl.Rule(rainfall['medium'] & agriculture['low']    & industry['medium'], scarcity['medium']),
    ctrl.Rule(rainfall['medium'] & agriculture['low']    & industry['high'],   scarcity['medium']),
    ctrl.Rule(rainfall['medium'] & agriculture['medium'] & industry['low'],    scarcity['medium']),
    ctrl.Rule(rainfall['medium'] & agriculture['medium'] & industry['medium'], scarcity['medium']),
    ctrl.Rule(rainfall['medium'] & agriculture['medium'] & industry['high'],   scarcity['high']),
    ctrl.Rule(rainfall['medium'] & agriculture['high']   & industry['low'],    scarcity['medium']),
    ctrl.Rule(rainfall['medium'] & agriculture['high']   & industry['medium'], scarcity['high']),
    ctrl.Rule(rainfall['medium'] & agriculture['high']   & industry['high'],   scarcity['high']),

    # Rainfall = HIGH
    ctrl.Rule(rainfall['high'] & agriculture['low']    & industry['low'],    scarcity['low']),
    ctrl.Rule(rainfall['high'] & agriculture['low']    & industry['medium'], scarcity['low']),
    ctrl.Rule(rainfall['high'] & agriculture['low']    & industry['high'],   scarcity['medium']),
    ctrl.Rule(rainfall['high'] & agriculture['medium'] & industry['low'],    scarcity['low']),
    ctrl.Rule(rainfall['high'] & agriculture['medium'] & industry['medium'], scarcity['low']),
    ctrl.Rule(rainfall['high'] & agriculture['medium'] & industry['high'],   scarcity['medium']),
    ctrl.Rule(rainfall['high'] & agriculture['high']   & industry['low'],    scarcity['medium']),
    ctrl.Rule(rainfall['high'] & agriculture['high']   & industry['medium'], scarcity['medium']),
    ctrl.Rule(rainfall['high'] & agriculture['high']   & industry['high'],   scarcity['medium']),
]

scarcity_ctrl = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(scarcity_ctrl)

# -----------------------------
# STEP 6: Example simulation + Hypothesis/Conclusion
# -----------------------------
sample = country_df.iloc[0]
try:
    rain_val = sample['Rainfall Impact (Annual Precipitation in mm)']
    ag_val   = sample['Agricultural Water Use (%)']
    ind_val  = sample['Industrial Water Use (%)']

    sim.input['rainfall'] = rain_val
    sim.input['agriculture'] = ag_val
    sim.input['industry'] = ind_val
    sim.compute()
    sc = sim.output['scarcity']
    print(f"\nComputed Water Scarcity Level: {sc:.2f}")

    # --- Hypothesis/Conclusion block (linguistic summary) ---
    def _ling_label(var, x):
        scores = {name: fuzz.interp_membership(var.universe, term.mf, x)
                  for name, term in var.terms.items()}
        label = max(scores, key=scores.get)
        return label, scores

    rain_lab, _ = _ling_label(rainfall,    rain_val)
    ag_lab, ag_mu = _ling_label(agriculture, ag_val)
    ind_lab, ind_mu = _ling_label(industry,    ind_val)
    scar_lab, _ = _ling_label(scarcity,    sc)

    ag_pressure  = ag_mu.get('high', 0.0)
    ind_pressure = ind_mu.get('high', 0.0)

    if ag_pressure > ind_pressure + 0.10:
        driver = "Agriculture demand dominates"
        rec = "Allocate more to AGRICULTURE or prioritize conservation there."
    elif ind_pressure > ag_pressure + 0.10:
        driver = "Industry demand dominates"
        rec = "Allocate more to INDUSTRY or tighten industrial use."
    else:
        driver = "Demand pressure is balanced"
        rec = "Balanced allocation; use policy priorities to decide."

    print("Hypothesis/Conclusion:")
    print(f"  With {rain_lab} rainfall, {ag_lab} agricultural use, and {ind_lab} industrial use,")
    print(f"  the model predicts **{scar_lab} scarcity** at ≈ {sc:.1f}/100.")
    print(f"  Driver: {driver}. {rec}")
    # --------------------------------------------------------

except KeyError as e:
    print(f"KeyError: {e} — please check the column names above.")

# -----------------------------
# STEP 7: 2×2 Membership Plots
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.ravel()

def plot_var(ax, var, title):
    x = var.universe
    for name, term in var.terms.items():
        ax.plot(x, term.mf, linewidth=1.5, label=name.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Value')
    ax.set_ylabel('Membership Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_var(axs[0], rainfall, 'Rainfall Impact (mm)')
plot_var(axs[1], agriculture, 'Agricultural Water Use (%)')
plot_var(axs[2], industry, 'Industrial Water Use (%)')
plot_var(axs[3], scarcity, 'Water Scarcity Level')

plt.tight_layout()
plt.show()

# -----------------------------
# STEP 8: 3D Surface Comparisons (side-by-side)
# -----------------------------
rain_vals = np.linspace(rain_min, rain_max, 30)
agri_vals = np.linspace(agri_min, agri_max, 30)
ind_vals  = np.linspace(ind_min, ind_max, 30)

# First surface: Rainfall × Agriculture (Industry fixed at min)
x1, y1 = np.meshgrid(rain_vals, agri_vals)
z1 = np.zeros_like(x1)

for i in range(30):
    for j in range(30):
        sim_temp = ctrl.ControlSystemSimulation(scarcity_ctrl)
        sim_temp.input['rainfall'] = x1[i, j]
        sim_temp.input['agriculture'] = y1[i, j]
        sim_temp.input['industry'] = ind_min
        try:
            sim_temp.compute()
            z1[i, j] = sim_temp.output.get('scarcity', np.nan)
        except Exception:
            z1[i, j] = np.nan

z1 = np.nan_to_num(z1, nan=0.0)

# Second surface: Rainfall × Industry (Agriculture fixed at min)
x2, y2 = np.meshgrid(rain_vals, ind_vals)
z2 = np.zeros_like(x2)

for i in range(30):
    for j in range(30):
        sim_temp = ctrl.ControlSystemSimulation(scarcity_ctrl)
        sim_temp.input['rainfall'] = x2[i, j]
        sim_temp.input['industry'] = y2[i, j]
        sim_temp.input['agriculture'] = agri_min
        try:
            sim_temp.compute()
            z2[i, j] = sim_temp.output.get('scarcity', np.nan)
        except Exception:
            z2[i, j] = np.nan

z2 = np.nan_to_num(z2, nan=0.0)

# Combined figure
fig = plt.figure(figsize=(14, 6))

# Left plot: Rainfall vs Agriculture
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(x1, y1, z1, cmap='viridis', edgecolor='none')
ax1.set_xlabel('Rainfall Impact (mm)')
ax1.set_ylabel('Agricultural Water Use (%)')
ax1.set_zlabel('Water Scarcity Level')
ax1.set_title('Rainfall vs Agriculture vs Scarcity')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Right plot: Rainfall vs Industry
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(x2, y2, z2, cmap='plasma', edgecolor='none')
ax2.set_xlabel('Rainfall Impact (mm)')
ax2.set_ylabel('Industrial Water Use (%)')
ax2.set_zlabel('Water Scarcity Level')
ax2.set_title('Rainfall vs Industry vs Scarcity')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
