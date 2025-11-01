import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv('dataset/cleaned_global_water_consumption.csv')

# ------------------------------------------------------------
# Define fuzzy inputs and outputs based on dataset ranges
# ------------------------------------------------------------
agriculture = ctrl.Antecedent(np.arange(28.9, 66.6, 0.1), 'Agricultural Water Use (%)')
industry    = ctrl.Antecedent(np.arange(13.2, 43.6, 0.1), 'Industrial Water Use (%)')
rainfall    = ctrl.Antecedent(np.arange(700, 2534, 1), 'Rainfall Impact (mm)')
depletion   = ctrl.Antecedent(np.arange(1.3, 4.33, 0.01), 'Groundwater Depletion Rate (%)')
scarcity    = ctrl.Consequent(np.arange(0, 11, 1), 'Water Scarcity Level')

# ------------------------------------------------------------
# Define membership functions
# ------------------------------------------------------------
# Agricultural Water Use
agriculture['low']    = mf.trimf(agriculture.universe, [28.9, 30, 40])
agriculture['medium'] = mf.trimf(agriculture.universe, [35, 45, 55])
agriculture['high']   = mf.trimf(agriculture.universe, [50, 60, 66.6])

# Industrial Water Use
industry['low']    = mf.trimf(industry.universe, [13.2, 15, 25])
industry['medium'] = mf.trimf(industry.universe, [20, 28, 35])
industry['high']   = mf.trimf(industry.universe, [30, 40, 43.6])

# Rainfall Impact
rainfall['low']    = mf.trimf(rainfall.universe, [700, 900, 1200])
rainfall['medium'] = mf.trimf(rainfall.universe, [1100, 1600, 2000])
rainfall['high']   = mf.trimf(rainfall.universe, [1800, 2300, 2534])

# Groundwater Depletion Rate
depletion['low']    = mf.trimf(depletion.universe, [1.3, 1.8, 2.3])
depletion['medium'] = mf.trimf(depletion.universe, [2.0, 2.7, 3.3])
depletion['high']   = mf.trimf(depletion.universe, [3.0, 3.7, 4.33])

# Water Scarcity Level
scarcity['low']    = mf.trimf(scarcity.universe, [0, 2, 4])
scarcity['medium'] = mf.trimf(scarcity.universe, [3, 5, 7])
scarcity['high']   = mf.trimf(scarcity.universe, [6, 8, 10])

# ------------------------------------------------------------
# Define fuzzy rules
# ------------------------------------------------------------
rule1 = ctrl.Rule(rainfall['low'] & depletion['high'], scarcity['high'])
rule2 = ctrl.Rule(rainfall['medium'] & depletion['medium'], scarcity['medium'])
rule3 = ctrl.Rule(rainfall['high'] & depletion['low'], scarcity['low'])
rule4 = ctrl.Rule(agriculture['high'] & industry['high'], scarcity['high'])
rule5 = ctrl.Rule(agriculture['low'] & rainfall['high'], scarcity['low'])
rule6 = ctrl.Rule(depletion['medium'] & rainfall['low'], scarcity['medium'])

# Catch-all rule for cases with no strong activation
rule7 = ctrl.Rule(~rainfall['low'] | ~depletion['high'], scarcity['medium'])

# ------------------------------------------------------------
# Control system and simulation
# ------------------------------------------------------------
scarcity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
scarcity_sim  = ctrl.ControlSystemSimulation(scarcity_ctrl)

# Example input values
scarcity_sim.input['Agricultural Water Use (%)'] = 55
scarcity_sim.input['Industrial Water Use (%)'] = 30
scarcity_sim.input['Rainfall Impact (mm)'] = 1200
scarcity_sim.input['Groundwater Depletion Rate (%)'] = 3.5

# Compute fuzzy result
scarcity_sim.compute()
print(f"Predicted Water Scarcity Level: {scarcity_sim.output['Water Scarcity Level']:.2f}")

# ------------------------------------------------------------
# Visualize membership functions (optional)
# ------------------------------------------------------------
agriculture.view()
industry.view()
rainfall.view()
depletion.view()
scarcity.view()

# ------------------------------------------------------------
# 3D Fuzzy surface visualization (Rainfall vs Depletion)
# ------------------------------------------------------------
rainfall_vals  = np.arange(700, 2534, 50)
depletion_vals = np.arange(1.3, 4.33, 0.05)
x, y = np.meshgrid(rainfall_vals, depletion_vals)
z = np.zeros_like(x)

for i in range(len(rainfall_vals)):
	for j in range(len(depletion_vals)):
		sim = ctrl.ControlSystemSimulation(scarcity_ctrl)
		sim.input['Rainfall Impact (mm)'] = x[j, i]
		sim.input['Groundwater Depletion Rate (%)'] = y[j, i]
		sim.input['Agricultural Water Use (%)'] = 45
		sim.input['Industrial Water Use (%)'] = 25

		try:
			sim.compute()
			z[j, i] = sim.output.get('Water Scarcity Level', np.nan)
		except Exception:
			z[j, i] = np.nan

# 3D Surface Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('Rainfall Impact (mm)')
ax.set_ylabel('Groundwater Depletion Rate (%)')
ax.set_zlabel('Predicted Water Scarcity Level')
ax.set_title('3D Fuzzy Surface: Rainfall vs Depletion vs Scarcity')
plt.show()
