import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Insurance Survival Analytics Suite", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.risk-score-high { color: #ff4444; font-weight: bold; }
.risk-score-medium { color: #ff8800; font-weight: bold; }
.risk-score-low { color: #44aa44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Insurance Survival Analytics Suite</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
    Advanced survival analysis for insurance risk assessment using real-world mortality patterns and comprehensive risk modeling.
    </p>
</div>
""", unsafe_allow_html=True)

def calculate_risk_scores(ages, genders, smokers, bmis, diabetes, heart_disease, 
                         hypertension, family_history, exercise_freq, alcohol_consumption, occupations):
    """Calculate comprehensive risk scores using actuarial methods"""
    
    scores = np.zeros(len(ages))
    
    # Age factor (exponential increase after 45)
    scores += np.where(ages < 45, (ages - 20) * 0.5, (ages - 20) * 1.2)
    
    # Gender factor (males typically higher risk)
    scores += np.where(np.array(genders) == 'Male', 5, 0)
    
    # Smoking (major risk factor)
    scores += smokers * 15
    
    # BMI (U-shaped curve - both underweight and obese are risky)
    bmi_risk = np.where(bmis < 18.5, 8, 
                       np.where(bmis > 30, (bmis - 30) * 2, 
                               np.where(bmis > 25, (bmis - 25) * 0.5, 0)))
    scores += bmi_risk
    
    # Medical conditions
    scores += diabetes * 10
    scores += heart_disease * 12
    scores += hypertension * 6
    scores += family_history * 4
    
    # Exercise frequency (protective factor)
    exercise_map = {'Never': 8, 'Rarely': 4, 'Sometimes': 0, 'Often': -2, 'Daily': -4}
    exercise_scores = [exercise_map[freq] for freq in exercise_freq]
    scores += exercise_scores
    
    # Alcohol consumption
    alcohol_map = {'None': 0, 'Light': -1, 'Moderate': 2, 'Heavy': 8}
    alcohol_scores = [alcohol_map[level] for level in alcohol_consumption]
    scores += alcohol_scores
    
    # Occupation risk
    occupation_map = {'Low Risk': 0, 'Medium Risk': 3, 'High Risk': 8}
    occupation_scores = [occupation_map[risk] for risk in occupations]
    scores += occupation_scores
    
    return np.clip(scores, 0, 100)

def categorize_risk(scores):
    """Categorize risk scores into risk levels"""
    return np.where(scores < 30, 'Low Risk',
                   np.where(scores < 60, 'Medium Risk', 'High Risk'))

# =============================================================================
# SURVIVAL ANALYSIS FUNCTIONS
# =============================================================================

def kaplan_meier_estimator(durations, events):
    """Enhanced Kaplan-Meier estimator with confidence intervals"""
    durations = np.array(durations)
    events = np.array(events)
    
    if len(durations) == 0:
        return [], [], [], []
    
    # Sort by duration
    sorted_indices = np.argsort(durations)
    durations = durations[sorted_indices]
    events = events[sorted_indices]
    
    # Get unique time points
    unique_times = np.unique(durations)
    
    survival_times = [0]
    survival_probs = [1.0]
    confidence_lower = [1.0]
    confidence_upper = [1.0]
    
    n_total = len(durations)
    
    for t in unique_times:
        # Number at risk at time t
        at_risk = np.sum(durations >= t)
        
        # Number of events at time t
        events_at_t = np.sum((durations == t) & (events == 1))
        
        if at_risk == 0:
            break
            
        # Calculate survival probability
        survival_prob = survival_probs[-1] * (1 - events_at_t / at_risk)
        
        # Calculate confidence interval (Greenwood's formula)
        if survival_prob > 0:
            var_log_s = 0
            for i, time_point in enumerate(survival_times[1:], 1):
                at_risk_i = np.sum(durations >= time_point)
                events_i = np.sum((durations == time_point) & (events == 1))
                if at_risk_i > 0 and events_i > 0:
                    var_log_s += events_i / (at_risk_i * (at_risk_i - events_i))
            
            if at_risk > events_at_t:
                var_log_s += events_at_t / (at_risk * (at_risk - events_at_t))
            
            se_log_s = np.sqrt(var_log_s)
            
            # 95% confidence interval
            factor = 1.96 * se_log_s
            conf_lower = survival_prob ** np.exp(factor)
            conf_upper = survival_prob ** np.exp(-factor)
            
            confidence_lower.append(max(0, conf_lower))
            confidence_upper.append(min(1, conf_upper))
        else:
            confidence_lower.append(0)
            confidence_upper.append(0)
        
        survival_times.append(t)
        survival_probs.append(survival_prob)
    
    return survival_times, survival_probs, confidence_lower, confidence_upper

def nelson_aalen_estimator(durations, events):
    """Nelson-Aalen cumulative hazard estimator"""
    durations = np.array(durations)
    events = np.array(events)
    
    if len(durations) == 0:
        return [], []
    
    sorted_indices = np.argsort(durations)
    durations = durations[sorted_indices]
    events = events[sorted_indices]
    
    unique_times = np.unique(durations)
    
    hazard_times = [0]
    cumulative_hazards = [0]
    
    for t in unique_times:
        at_risk = np.sum(durations >= t)
        events_at_t = np.sum((durations == t) & (events == 1))
        
        if at_risk == 0:
            break
            
        hazard_increment = events_at_t / at_risk
        cumulative_hazard = cumulative_hazards[-1] + hazard_increment
        
        hazard_times.append(t)
        cumulative_hazards.append(cumulative_hazard)
    
    return hazard_times, cumulative_hazards

# =============================================================================
# MAIN APPLICATION
# =============================================================================
@st.cache_data
def load_real_data():
    df = pd.read_csv(r"C:\Users\user\Desktop\Survivalsuite\data\insurance_real_data.csv")

    df.rename(columns={
        'age': 'Age',
        'sex': 'Gender',
        'bmi': 'BMI',
        'smoker': 'Smoker',
        'children': 'Children',
        'charges': 'Policy_Amount'
    }, inplace=True)

    df['Gender'] = df['Gender'].str.capitalize()
    df['Smoker'] = df['Smoker'].map({'yes': 1, 'no': 0})

    # Add dummy values for missing features
    df['Diabetes'] = 0
    df['Heart_Disease'] = 0
    df['Hypertension'] = 0
    df['Family_History'] = 0
    df['Exercise_Frequency'] = ['Sometimes'] * len(df)
    df['Alcohol_Consumption'] = ['Moderate'] * len(df)
    df['Occupation_Risk'] = ['Medium Risk'] * len(df)

    # Add synthetic duration and event columns
    np.random.seed(42)
    df['Duration_Months'] = np.random.randint(6, 120, size=len(df))
    df['Event_Occurred'] = np.random.binomial(1, 0.3, size=len(df))

    # Apply risk model
    df['Risk_Score'] = calculate_risk_scores(
        df['Age'], df['Gender'], df['Smoker'],
        df['BMI'], df['Diabetes'], df['Heart_Disease'],
        df['Hypertension'], df['Family_History'],
        df['Exercise_Frequency'], df['Alcohol_Consumption'], df['Occupation_Risk']
    )
    df['Risk_Category'] = categorize_risk(df['Risk_Score'])
    df['Policy_Type'] = 'Term'  # placeholder if needed

    return df

df = load_real_data()
# Sidebar filters
st.sidebar.header("🎛️ Risk Analysis Filters")

# Risk-based filtering
risk_categories = st.sidebar.multiselect(
    "Risk Categories", 
    options=df['Risk_Category'].unique(), 
    default=df['Risk_Category'].unique()
)

# Demographics
age_range = st.sidebar.slider(
    "Age Range", 
    min_value=int(df['Age'].min()), 
    max_value=int(df['Age'].max()), 
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

genders = st.sidebar.multiselect(
    "Gender", 
    options=df['Gender'].unique(), 
    default=df['Gender'].unique()
)

# Apply filters
filtered_df = df[
    (df['Risk_Category'].isin(risk_categories)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Gender'].isin(genders)) 
]

# =============================================================================
# DASHBOARD LAYOUT
# =============================================================================

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Policies", f"{len(filtered_df):,}")

with col2:
    event_rate = filtered_df['Event_Occurred'].mean()
    st.metric("Event Rate", f"{event_rate:.1%}")

with col3:
    avg_duration = filtered_df['Duration_Months'].mean()
    st.metric("Avg Duration", f"{avg_duration:.1f} months")

with col4:
    avg_risk_score = filtered_df['Risk_Score'].mean()
    risk_color = "🔴" if avg_risk_score > 60 else "🟡" if avg_risk_score > 30 else "🟢"
    st.metric("Avg Risk Score", f"{avg_risk_score:.1f} {risk_color}")

# Data overview
st.subheader("📊 Portfolio Overview")

col1, col2 = st.columns(2)

with col1:
    # Risk distribution
    fig_risk = px.histogram(
        filtered_df, 
        x='Risk_Score', 
        color='Risk_Category',
        nbins=30,
        title="Risk Score Distribution"
    )
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    # Age vs Risk Score
    fig_age_risk = px.scatter(
        filtered_df, 
        x='Age', 
        y='Risk_Score',
        color='Risk_Category',
        size='Policy_Amount',
        title="Age vs Risk Score"
    )
    fig_age_risk.update_layout(height=400)
    st.plotly_chart(fig_age_risk, use_container_width=True)

# Survival Analysis
st.subheader("📈 Survival Analysis")

# Stratification options
stratify_options = ['Risk_Category', 'Gender', 'Smoker', 'Policy_Type', 'Age_Group']

# Create age groups for stratification
filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], 
                                 bins=[0, 35, 50, 65, 100], 
                                 labels=['Under 35', '35-50', '50-65', '65+'])

group_col = st.selectbox("Stratify survival curves by:", options=stratify_options)

# Create survival curves
fig_survival = go.Figure()

colors = px.colors.qualitative.Set1
for i, group in enumerate(filtered_df[group_col].unique()):
    if pd.isna(group):
        continue
        
    group_data = filtered_df[filtered_df[group_col] == group]
    
    if len(group_data) > 0:
        times, probs, conf_lower, conf_upper = kaplan_meier_estimator(
            group_data['Duration_Months'], 
            group_data['Event_Occurred']
        )
        
        color = colors[i % len(colors)]
        
        # Main survival curve
        fig_survival.add_trace(go.Scatter(
            x=times, 
            y=probs,
            mode='lines',
            name=str(group),
            line=dict(color=color, width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x} months<br>' +
                         'Survival Probability: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Confidence intervals
        fig_survival.add_trace(go.Scatter(
            x=times + times[::-1],
            y=conf_upper + conf_lower[::-1],
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{group} CI',
            hoverinfo='skip'
        ))

fig_survival.update_layout(
    title=f"Kaplan-Meier Survival Curves by {group_col}",
    xaxis_title="Time (months)",
    yaxis_title="Survival Probability",
    height=600,
    hovermode='x unified'
)

st.plotly_chart(fig_survival, use_container_width=True)

# Hazard Analysis
st.subheader("⚡ Hazard Analysis")

col1, col2 = st.columns(2)

with col1:
    # Cumulative hazard
    fig_hazard = go.Figure()
    
    for i, group in enumerate(filtered_df[group_col].unique()):
        if pd.isna(group):
            continue
            
        group_data = filtered_df[filtered_df[group_col] == group]
        
        if len(group_data) > 0:
            hazard_times, cumulative_hazards = nelson_aalen_estimator(
                group_data['Duration_Months'], 
                group_data['Event_Occurred']
            )
            
            fig_hazard.add_trace(go.Scatter(
                x=hazard_times,
                y=cumulative_hazards,
                mode='lines',
                name=str(group),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig_hazard.update_layout(
        title="Nelson-Aalen Cumulative Hazard",
        xaxis_title="Time (months)",
        yaxis_title="Cumulative Hazard",
        height=400
    )
    
    st.plotly_chart(fig_hazard, use_container_width=True)

with col2:
    # Risk factor analysis
    risk_factors = ['Age', 'BMI', 'Risk_Score']
    selected_factor = st.selectbox("Analyze risk factor:", risk_factors)
    
    # Create risk factor plot
    fig_factor = px.box(
        filtered_df,
        x='Risk_Category',
        y=selected_factor,
        title=f"{selected_factor} by Risk Category"
    )
    fig_factor.update_layout(height=400)
    st.plotly_chart(fig_factor, use_container_width=True)

# Detailed Data Table
st.subheader("📋 Detailed Policy Data")

# Display options
show_columns = st.multiselect(
    "Select columns to display:",
    options=df.columns.tolist(),
    default=['Age', 'Gender', 'Risk_Score', 'Risk_Category', 'Duration_Months', 'Event_Occurred']
)

if show_columns:
    display_df = filtered_df[show_columns].copy()
    
    # Add risk score styling
    def style_risk_score(val):
        if 'Risk_Score' in str(val):
            return val
        return val
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Export functionality
st.subheader("💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Export filtered data
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"insurance_survival_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    # Export summary statistics
    summary_stats = filtered_df.groupby('Risk_Category').agg({
        'Age': ['mean', 'std'],
        'Duration_Months': ['mean', 'std'],
        'Event_Occurred': ['mean', 'count'],
        'Risk_Score': ['mean', 'std']
    }).round(2)
    
    summary_csv = summary_stats.to_csv().encode('utf-8')
    st.download_button(
        label="📊 Download Summary Statistics (CSV)",
        data=summary_csv,
        file_name=f"survival_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 Insurance Survival Analytics Suite | Built with Streamlit | 
    Dataset: {n_policies:,} policies | Last updated: {timestamp}</p>
</div>
""".format(
    n_policies=len(df),
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
), unsafe_allow_html=True)