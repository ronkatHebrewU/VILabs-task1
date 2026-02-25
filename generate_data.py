import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

def generate_raw_data():
    """
    Generates synthetic datasets for the VILabs project.
    Returns:
        hcp_df (pd.DataFrame): Master data for HCPs.
        rx_df (pd.DataFrame): Prescription history.
        interventions_df (pd.DataFrame): Field interaction logs.
    """
    num_hcps = 3000

    # 1. GENERATE HCP MASTER DATA
    # This table represents the core profiles of 3,000 healthcare professionals[cite: 39, 40].
    hcp_df = pd.DataFrame({
        'hcp_id': np.arange(1000, 1000 + num_hcps),
        'specialty': np.random.choice(['Oncology', 'Hematology'], num_hcps, p=[0.7, 0.3]),
        'segment': np.random.choice(['High Potential', 'Medium Potential', 'Low Potential'], num_hcps, p=[0.2, 0.5, 0.3]),
        'territory_id': [f'Territory_{i:02d}' for i in np.random.randint(1, 11, num_hcps)]
    })

    # 2. GENERATE RX HISTORY DATA (Last 6 Months)
    # Captures prescription volume for both the client's drug and the competitor[cite: 39].
    months = ['2025-09', '2025-10', '2025-11', '2025-12', '2026-01', '2026-02']
    rx_records = []

    for hcp_id in hcp_df['hcp_id']:
        # Set a baseline monthly volume for each HCP
        base_vol = np.random.randint(5, 30)
        # Inject a "churn" signal into 15% of the HCPs for the Script Defense model.
        is_at_risk = np.random.random() < 0.15

        for i, month in enumerate(months):
            # If at risk, the client's scripts decrease while competitor scripts increase
            trend = (1 - (i * 0.15)) if is_at_risk else 1.0
            client_v = max(0, int(base_vol * trend + np.random.randint(-2, 3)))
            comp_v = max(0, int((35 - base_vol) + (i * 2 if is_at_risk else 0)))

            rx_records.append({
                'hcp_id': hcp_id,
                'month': month,
                'client_scripts': client_v,
                'competitor_scripts': comp_v
            })

    rx_df = pd.DataFrame(rx_records)

    # 3. GENERATE FIELD INTERVENTIONS (CRM DATA)
    # Includes a 'Notes' column to simulate unstructured data for AI Agent processing[cite: 41].
    interaction_types = ['Face-to-Face Visit', 'Remote Call', 'Email']
    notes_pool = [
        "HCP requested latest clinical trial data.",
        "Discussed side effect profile; HCP expressed no concerns.",
        "HCP mentioned competitor drug has a more favorable copay program.",  # Switch signal
        "Sample drops completed. Staff requested more brochures.",
        "HCP complained about prior authorization hurdles for our drug.",
        "Routine follow-up. HCP is satisfied with patient outcomes.",
        "HCP stated they are trial-running a competitor product for new patients."  # High risk signal
    ]

    intervention_records = []
    for hcp_id in hcp_df['hcp_id']:
        # Random number of interactions (0 to 4 per year)
        for _ in range(np.random.randint(0, 5)):
            intervention_records.append({
                'hcp_id': hcp_id,
                'date': f"2025-{np.random.randint(9, 13):02d}-{np.random.randint(1, 28):02d}",
                'interaction_type': np.random.choice(interaction_types),
                'notes': np.random.choice(notes_pool)
            })

    interventions_df = pd.DataFrame(intervention_records)
    
    return hcp_df, rx_df, interventions_df

if __name__ == "__main__":
    hcp, rx, interventions = generate_raw_data()
    # DISPLAY SAMPLE OUTPUTS
    print("--- HCP Master Sample ---")
    print(hcp.head(3))
    print("\n--- Rx History Sample ---")
    print(rx.head(3))
    print("\n--- Field Interventions Sample ---")
    print(interventions.head(3))