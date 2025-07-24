import yaml
import os

def load_config():
    """Load emission factors and assumptions from YAML config."""
    with open("config/assumptions.yaml", "r") as file:
        return yaml.safe_load(file)

def estimate_emissions(llm_result):
    """Estimate emissions based on LLM output."""
    try:
        quantity = llm_result["activity_data"]["estimated_quantity"]
        emission_factor = llm_result["emission_factor"]["value"]
        emissions_kg = quantity * emission_factor
        llm_result["emissions_kg"] = emissions_kg
        return llm_result
    except Exception as e:
        print(f"Error estimating emissions: {e}")
        return llm_result