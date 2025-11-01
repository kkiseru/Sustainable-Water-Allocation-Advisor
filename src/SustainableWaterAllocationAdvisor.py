import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ------------------------------------------------------------
# Load dataset using pandas
# ------------------------------------------------------------
df = pd.read_csv('dataset/cleaned_global_water_consumption.csv')

# Print one of the headers (for confirmation)
print("One of the headers is:", df.columns[0])

# Calculating the range of each variable (DO NOT REMOVE)
# print("\nFeature ranges:")
# print("Agricultural Water Use (%):", df['Agricultural Water Use (%)'].min(), "-", df['Agricultural Water Use (%)'].max())
# print("Industrial Water Use (%):", df['Industrial Water Use (%)'].min(), "-", df['Industrial Water Use (%)'].max())
# print("Rainfall Impact (Annual Precipitation in mm):", df['Rainfall Impact (Annual Precipitation in mm)'].min(), "-", df['Rainfall Impact (Annual Precipitation in mm)'].max())
# print("Groundwater Depletion Rate (%):", df['Groundwater Depletion Rate (%)'].min(), "-", df['Groundwater Depletion Rate (%)'].max())

# ------------------------------------------------------------
# Define fuzzy inputs and outputs based on dataset ranges
# ------------------------------------------------------------

# Input 1: Agricultural Water Use (%)
agriculture = ctrl.Antecedent(np.arange(28.9, 66.6, 0.1), 'Agricultural Water Use (%)')

# Input 2: Industrial Water Use (%)
industry = ctrl.Antecedent(np.arange(13.2, 43.6, 0.1), 'Industrial Water Use (%)')

# Input 3: Rainfall Impact (mm)
rainfall = ctrl.Antecedent(np.arange(700, 2534, 1), 'Rainfall Impact (mm)')

# Input 4: Groundwater Depletion Rate (%)
depletion = ctrl.Antecedent(np.arange(1.3, 4.33, 0.01), 'Groundwater Depletion Rate (%)')

# Output: Water Scarcity Level
scarcity = ctrl.Consequent(np.arange(0, 11, 1), 'Water Scarcity Level')
