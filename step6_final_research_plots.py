import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# -----------------------------
# Argument parser (CSV input)
# -----------------------------
parser = argparse.ArgumentParser(description="Scaling Analysis Plotter")
parser.add_argument(
    "--csv",
    type=str,
    required=True,
    help="Path to input CSV file"
)
args = parser.parse_args()

# -----------------------------
# Load CSV
# -----------------------------
if not os.path.exists(args.csv):
    raise FileNotFoundError(f"CSV file not found: {args.csv}")

df = pd.read_csv(args.csv)

# Expected columns:
# project_name, total_duration (sec), total_emissions (kg), total_energy (kWh)

# -----------------------------
# Clean labels
# -----------------------------
df['model_label'] = df['project_name'].str.replace('FineTune_', '', regex=False)

# -----------------------------
# Plot setup
# -----------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Scaling Analysis: Carbon, Energy, and Time', fontsize=16)

# -----------------------------
# Plot 1: Carbon Emissions
# -----------------------------
sns.barplot(
    ax=axes[0],
    x='model_label',
    y='total_emissions (kg)',
    data=df
)
axes[0].set_title('Carbon Emissions (kg CO₂)')
axes[0].set_ylabel('kg CO₂')
axes[0].set_xlabel('Model')

# -----------------------------
# Plot 2: Energy Consumption
# -----------------------------
sns.barplot(
    ax=axes[1],
    x='model_label',
    y='total_energy (kWh)',
    data=df
)
axes[1].set_title('Energy Consumed (kWh)')
axes[1].set_ylabel('kWh')
axes[1].set_xlabel('Model')

# -----------------------------
# Plot 3: Training Duration
# -----------------------------
sns.barplot(
    ax=axes[2],
    x='model_label',
    y='total_duration (sec)',
    data=df
)
axes[2].set_title('Training Duration (seconds)')
axes[2].set_ylabel('Seconds')
axes[2].set_xlabel('Model')

# -----------------------------
# Final layout & save
# -----------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("scaling_analysis.png", dpi=300)
plt.show()

# -----------------------------
# Print summary
# -----------------------------
print("\nInput CSV Summary:\n")
print(df.to_string(index=False))
