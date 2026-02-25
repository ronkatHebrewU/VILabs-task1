import generate_data
import data_processor
import thinking_layer

# --- CONSTANTS ---
OUTPUT_FILENAME = "field_action_plan.csv"
TOP_N_RECOMMENDATIONS = 10

def run_pipeline():
    """
    Executes the full 'Listen-Think-Act' pipeline.
    """
    print("--- Starting VILabs Pipeline ---")

    # 1. LISTEN: Generate raw data
    print("Step 1: Generating raw data...")
    hcp_df, rx_df, interventions_df = generate_data.generate_raw_data()
    print(f"Generated {len(hcp_df)} HCPs, {len(rx_df)} Rx records, {len(interventions_df)} Interventions.")

    # 2. THINK (Part 1): Create advanced feature matrix
    print("Step 2: Processing data and creating feature matrix...")
    advanced_fm = data_processor.create_advanced_feature_matrix(hcp_df, rx_df, interventions_df)
    print(f"Feature matrix created with shape: {advanced_fm.shape}")

    # 3. THINK (Part 2): Run Uplift and Propensity modeling
    print("Step 3: Running AI models (Uplift & Propensity)...")
    recommendations_df = thinking_layer.execute_advanced_thinking_layer(advanced_fm)
    print(f"Modeling complete. Predictions generated for {len(recommendations_df)} HCPs.")

    # 4. ACT: Filter and prioritize recommendations
    print("Step 4: Generating actionable recommendations...")
    
    # Filter for 'Persuadable' HCPs
    persuadable_hcps = recommendations_df[recommendations_df['uplift_segment'] == "Persuadable (Target)"]
    
    # Sort by risk (T1) and uplift score
    # We want high risk (propensity_T1) and high uplift
    top_recommendations = persuadable_hcps.sort_values(by=['propensity_T1', 'uplift_score'], ascending=False).head(TOP_N_RECOMMENDATIONS)

    # Display Top 10 Recommendations
    print(f"\n--- Top {TOP_N_RECOMMENDATIONS} High-Impact Script Defense Recommendations ---")
    cols_to_show = ['hcp_id', 'uplift_score', 'propensity_T1', 'ideal_visit_type']
    print(top_recommendations[cols_to_show].to_string(index=False))

    # Export to CSV
    print(f"\nExporting results to {OUTPUT_FILENAME}...")
    top_recommendations.to_csv(OUTPUT_FILENAME, index=False)
    print("Pipeline execution finished successfully.")

if __name__ == "__main__":
    run_pipeline()