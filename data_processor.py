import pandas as pd
import numpy as np

# --- GLOBAL THRESHOLDS & CONSTANTS ---
MIN_INTERACTIONS_FOR_TAG = 2  # Minimum calls to be considered "Interacted"
RETAINED_SHARE_THRESHOLD = 0.8  # Brand share >= 80%
CHURNED_SHARE_THRESHOLD = 0.2  # Brand share < 20%
HIGH_VOLUME_THRESHOLD = 15  # Total scripts/month to be "High Volume"
LOW_VOLUME_THRESHOLD = 2  # Total scripts/month below which is "Inactive"
DEFAULT_RISK_SENTIMENT = 0.5  # Baseline sentiment score
RISK_BOOST = 0.4  # Increase in risk score for flagged keywords


def create_advanced_feature_matrix(hcp_df, rx_df, interventions_df):
    """
    Consolidates raw data and applies the 'Initial Thought Layer' tagging.
    """

    # 1. RX DYNAMICS & OUTCOME CALCULATION
    # Pivot to get the most recent month data
    rx_pivot = rx_df.pivot(index='hcp_id', columns='month', values=['client_scripts', 'competitor_scripts'])
    last_month = '2026-02'

    client_v = rx_pivot['client_scripts'][last_month]
    comp_v = rx_pivot['competitor_scripts'][last_month]
    total_v = client_v + comp_v
    market_share = client_v / total_v.replace(0, 1)  # Handling zero-volume cases

    rx_features = pd.DataFrame(index=hcp_df['hcp_id'])
    rx_features['current_total_vol'] = total_v
    rx_features['current_market_share'] = market_share

    # 2. FIELD INTERVENTIONS & AGENT INFERENCE
    # Simulation of an AI Agent inferring the Ideal Visit Type from unstructured data
    def agent_infer_ideal_visit(group):
        notes_blob = " ".join(group['notes'].fillna("").astype(str)).lower()
        if "competitor" in notes_blob or "trial-running" in notes_blob:
            return "Competitive Positioning (Face-to-Face)"
        elif "trial data" in notes_blob or "side effect" in notes_blob:
            return "Clinical/MSL Deep-Dive"
        elif "authorization" in notes_blob or "copay" in notes_blob:
            return "Market Access & Patient Support"
        return "Standard Clinical Update (Remote)"

    # Calculate risk score based on keywords
    def calculate_risk_score(group):
        notes_blob = " ".join(group['notes'].fillna("").astype(str)).lower()
        risk = DEFAULT_RISK_SENTIMENT
        if "competitor" in notes_blob or "trial-running" in notes_blob or "complained" in notes_blob:
            risk += RISK_BOOST
        return min(risk, 1.0)

    # Aggregate interaction counts and sentiment
    int_agg = interventions_df.groupby('hcp_id').agg({
        'interaction_type': 'count'
    }).rename(columns={'interaction_type': 'call_count'})

    # Apply Agent Inference
    visit_recommendations = interventions_df.groupby('hcp_id').apply(agent_infer_ideal_visit).to_frame(
        'ideal_visit_type')
    
    # Apply Risk Score Calculation
    risk_scores = interventions_df.groupby('hcp_id').apply(calculate_risk_score).to_frame('risk_score')

    int_features = int_agg.join(visit_recommendations).join(risk_scores)

    # 3. MERGING INTO FEATURE MATRIX
    # Use left_on and right_index because rx_features and int_features are indexed by hcp_id
    fm = hcp_df.merge(rx_features, left_on='hcp_id', right_index=True, how='left') \
        .merge(int_features, left_on='hcp_id', right_index=True, how='left')

    # Fill missing values for HCPs with no recent activity
    fm['call_count'] = fm['call_count'].fillna(0)
    fm['ideal_visit_type'] = fm['ideal_visit_type'].fillna("Initial Awareness Outreach")
    fm['risk_score'] = fm['risk_score'].fillna(DEFAULT_RISK_SENTIMENT)
    fm['current_total_vol'] = fm['current_total_vol'].fillna(0)
    fm['current_market_share'] = fm['current_market_share'].fillna(0)

    # 4. INITIAL THOUGHT LAYER (TAGGING)
    # Interaction Tagging
    fm['interaction_tag'] = np.where(fm['call_count'] > MIN_INTERACTIONS_FOR_TAG, 'Interacted', 'Not Interacted')

    # Outcome Tagging
    is_retained = (fm['current_market_share'] >= RETAINED_SHARE_THRESHOLD) & \
                  (fm['current_total_vol'] > HIGH_VOLUME_THRESHOLD)

    is_churned = (fm['current_market_share'] < CHURNED_SHARE_THRESHOLD) | \
                 (fm['current_total_vol'] < LOW_VOLUME_THRESHOLD)

    fm['outcome_tag'] = 'Unknown/Intermediate'
    fm.loc[is_retained, 'outcome_tag'] = 'Retained'
    fm.loc[is_churned, 'outcome_tag'] = 'Churned'

    return fm

if __name__ == "__main__":
    # Dummy execution for testing
    pass