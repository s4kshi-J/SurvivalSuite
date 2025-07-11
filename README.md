# SurvivalSuite
SurvivalSuite is an interactive survival analysis dashboard built with synthetic life insurance data.

## features
- Kaplan-Meier survival curves stratified by demographic/policy variables
- Cox Proportional Hazards regression
- Downloadable filtered dataset and model summary
- Interactive filters with live charts

# Custom Kaplan-Meier Estimator Function
def kaplan_meier_estimator(durations, events):
    durations = np.array(durations)
    events = np.array(events)
    sorted_indices = np.argsort(durations)
    durations = durations[sorted_indices]
    events = events[sorted_indices]
    
    unique_times = np.unique(durations)
    at_risk = len(durations)
    survival_prob = 1.0
    survival_times = []
    survival_probs = []

    for t in unique_times:
        d = np.sum((durations == t) & (events == 1))
        n = at_risk
        if n == 0:
            break
        survival_prob *= (1 - d / n)
        survival_times.append(t)
        survival_probs.append(survival_prob)
        at_risk -= np.sum(durations == t)
    
    return survival_times, survival_probs

## How to Run
Make sure you have Python 3.8–3.11 installed.
```bash
pip install -r requirements.txt
streamlit run src/app.py
