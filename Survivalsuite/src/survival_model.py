import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

df= pd.read_csv("data/synthetic_life_data.csv")
kmf= KaplanMeierFitter()

plt.figure(figsize=(10,6))
plt.title("Kaplan-Meier Survival curve by smoking status")
plt.xlabel("Time (Months)")
plt.ylabel("survival probability")
plt.grid(True)

for label, grouped_df in df.groupby("Smoker"):
    kmf.fit(grouped_df["Duration"],event_observed=grouped_df["Event"],label=f"smoker:{label}")
    kmf.plot(ci_show=False)

plt.tight_layout()
plt.savefig("data/kmf_smoking_survival.png")
plt.show()
