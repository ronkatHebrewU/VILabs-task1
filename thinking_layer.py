import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- CONSTANTS ---
UPLIFT_HIGH_THRESHOLD = 0.05  # Minimum delta to be a 'Persuadable'
PROPENSITY_HIGH_THRESHOLD = 0.7  # Score above which risk is critical
RANDOM_STATE = 42


def execute_advanced_thinking_layer(fm):
    """
    Implements Two-Model Uplift and Windowed Propensity for Script Defense.
    """
    # --- 1. PREPARE TRAINING DATA ---
    # We only train on HCPs where we already 'know' the outcome (Retained or Churned)
    train_df = fm[fm['outcome_tag'] != 'Unknown/Intermediate'].copy()
    predict_df = fm[fm['outcome_tag'] == 'Unknown/Intermediate'].copy()

    # Feature selection
    features = ['current_market_share', 'current_total_vol', 'risk_score']

    # Encode target: 1 for Retained, 0 for Churned
    y_train = (train_df['outcome_tag'] == 'Retained').astype(int)
    X_train = train_df[features]

    # --- 2. UPLIFT MODELING (TWO-MODEL APPROACH) ---
    # Model 1: Treatment (Interacted)
    treatment_mask = train_df['interaction_tag'] == 'Interacted'
    model_t = LogisticRegression(random_state=RANDOM_STATE)
    model_t.fit(X_train[treatment_mask], y_train[treatment_mask])

    # Model 2: Control (Not Interacted)
    control_mask = train_df['interaction_tag'] == 'Not Interacted'
    model_c = LogisticRegression(random_state=RANDOM_STATE)
    model_c.fit(X_train[control_mask], y_train[control_mask])

    # Inference on Unknowns
    X_pred = predict_df[features]
    p_treat = model_t.predict_proba(X_pred)[:, 1]  # P(Retained | Treatment)
    p_cont = model_c.predict_proba(X_pred)[:, 1]  # P(Retained | Control)

    predict_df['uplift_score'] = p_treat - p_cont

    # Classification into Uplift Quadrants
    def classify_uplift(row):
        if row['uplift_score'] > UPLIFT_HIGH_THRESHOLD:
            return "Persuadable (Target)"
        elif row['uplift_score'] < -UPLIFT_HIGH_THRESHOLD:
            return "Sleeping Dog (Avoid)"
        elif p_treat[predict_df.index.get_loc(row.name)] > 0.5:
            return "Sure Thing (Don't Waste Resources)"
        return "Lost Cause (No Impact Possible)"

    predict_df['uplift_segment'] = predict_df.apply(classify_uplift, axis=1)

    # --- 3. WINDOWED PROPENSITY MODELING ---
    # Simulating probability of churn across T+1, T+2, T+3
    # In a real scenario, this would be trained on historical time-lagged data.
    prop_model = LogisticRegression(random_state=RANDOM_STATE)
    prop_model.fit(X_train, 1 - y_train)  # Training to predict Churn (0 -> 1)

    base_prob = prop_model.predict_proba(X_pred)[:, 1]

    predict_df['propensity_T1'] = base_prob
    predict_df['propensity_T2'] = np.clip(base_prob * 1.2, 0, 1)  # Increasing risk over time
    predict_df['propensity_T3'] = np.clip(base_prob * 1.5, 0, 1)

    return predict_df

if __name__ == "__main__":
    # Dummy execution for testing
    pass