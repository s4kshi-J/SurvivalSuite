import pandas as pd
import numpy as np

def generate_life_data(n=2000, seed=42):
    np.random.seed(seed)

    age = np.random.normal(loc=45, scale=12, size=n).astype(int)
    age = np.clip(age, 18, 90)
    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.48, 0.52])
    smoker = np.random.choice(['Yes', 'No'], size=n, p=[0.25, 0.75])
    income_bracket = np.random.choice(['Low', 'Middle', 'High'], size=n, p=[0.3, 0.5, 0.2])
    policy_type = np.random.choice(['Term', 'Whole Life', 'Endowment'], size=n, p=[0.5, 0.3, 0.2])

    base_hazard = 0.005

    risk_score = (
        0.02 * (age - 40) +
        0.3 * (smoker == 'Yes') +
        0.1 * (gender == 'Male') +
        0.2 * (policy_type == 'Term') +
        0.15 * (income_bracket == 'Low')
    )

    hazard = base_hazard * (1 + risk_score)
    survival_time = np.random.exponential(scale=1 / hazard)
    censoring_time = np.random.uniform(60, 120, size=n)

    event_observed = survival_time <= censoring_time
    observed_time = np.minimum(survival_time, censoring_time)

    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Smoker': smoker,
        'IncomeBracket': income_bracket,
        'PolicyType': policy_type,
        'RiskScore': risk_score.round(2),
        'Duration': observed_time.round(1),
        'Event': event_observed.astype(int)
    })

    return df

if __name__ == "__main__":
    df = generate_life_data()
    df.to_csv("data/synthetic_life_data.csv", index=False)
    print("Data generated and saved to 'data/synthetic_life_data.csv'")
