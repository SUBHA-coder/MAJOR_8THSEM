import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
emissions_df = pd.read_csv("./qwen_results/emissions.csv")
rouge_df = pd.read_csv("qwen_rouge_results.csv")

# Clean emissions: CodeCarbon saves one row per model. 
# We need to sum them or pick the last ones.
emissions_summary = emissions_df.groupby('project_name').agg({
    'emissions': 'sum', 
    'duration': 'sum'
}).reset_index()

# Extract model name from project_name (e.g., 'FineTune_Qwen1.5-0.5B')
emissions_summary['model'] = emissions_summary['project_name'].str.replace('FineTune_', '')

# Merge
final_df = pd.merge(rouge_df, emissions_summary, on='model')

# 2. Plotting
plt.figure(figsize=(12, 5))

# Plot A: Performance vs Emissions
plt.subplot(1, 2, 1)
sns.scatterplot(data=final_df, x='emissions', y='rougeL', size='model', hue='model', legend=False)
for i in range(len(final_df)):
    plt.text(final_df.emissions[i], final_df.rougeL[i], final_df.model[i])
plt.title("Trade-off: ROUGE-L vs Carbon Emissions")
plt.xlabel("CO2 Emissions (kg)")
plt.ylabel("ROUGE-L Score")

# Plot B: Carbon Efficiency (Performance per unit of Carbon)
final_df['efficiency'] = final_df['rougeL'] / final_df['emissions']
plt.subplot(1, 2, 2)
sns.barplot(data=final_df, x='model', y='efficiency', palette='viridis')
plt.title("Carbon Efficiency (ROUGE-L / kg CO2)")
plt.ylabel("Score per kg of CO2")

plt.tight_layout()
plt.savefig("scaling_results_plot.png")
plt.show()

print("📈 Plot saved as 'scaling_results_plot.png'. Ready for your report!")