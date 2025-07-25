import pandas as pd

# Load the emission factor dataset
df = pd.read_csv("ghg-conversion-factors-2025-flat-format.csv")

def find_emission_factor(scope: str, fuel: str, unit: str):
    # Normalize inputs
    scope = scope.strip().title()
    fuel = fuel.strip().title()
    unit = unit.strip().lower()

    # Try to match by Scope, Fuel (Level 3), and Unit (UOM)
    match = df[
        (df["Scope"].str.strip().str.title() == scope) &
        (df["Level 3"].str.strip().str.title() == fuel) &
        (df["UOM"].str.strip().str.lower() == unit)
    ]

    if match.empty:
        return None

    # If multiple rows match, return the first one (or refine logic)
    row = match.iloc[0]

    return {
        "emission_factor_id": row["ID"],
        "emission_factor": float(row["GHG Conversion Factor 2025"]),
        "unit": row["UOM"],
        "ghg_unit": row["GHG/Unit"]
    }

# Example usage
agent_output = {
    "scope": "1",
    "fuel": "Butane",
    "unit": "tonnes"
}

factor = find_emission_factor(agent_output["scope"], agent_output["fuel"], agent_output["unit"])

print(factor)