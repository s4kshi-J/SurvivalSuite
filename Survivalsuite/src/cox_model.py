import pandas as pd
from lifelines import CoxPHFitter

df = pd.read_csv("data/synthetic_life_data.csv")
df = df.drop(columns=["RiskScore"])
df_encoded = pd.get_dummies(df, drop_first=True)

# Fit Cox PH Model
cox = CoxPHFitter()
cox.fit(df_encoded, duration_col="Duration", event_col="Event")

cox.print_summary()
cox.summary.to_csv("data/cox_model_summary.csv")
print("\n Cox summary saved to 'data/cox_model_summary.csv'")
