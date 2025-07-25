import boto3
import json
import os
from dotenv import load_dotenv
from emission_logic import load_config

# Load environment variables
load_dotenv()

def classify_and_estimate(item):
    """Classify line item and estimate emissions using AWS Bedrock (Claude)."""
    config = load_config()
    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    
    prompt = f"""  
    You are an ESG emissions expert. Given the following invoice line item and configuration, classify it into a GHG category, determine its scope, select the appropriate estimation method, and estimate emissions.

    Line Item:
    - Description: {item['description']}
    - Amount Spent (INR): {item['amount_spent_inr']}
    - Vendor: {item['vendor']}
    - Date: {item['date']}

    Configuration:
    {json.dumps(config, indent=2)}

    Return a JSON object with:
    - category: GHG category (e.g., Business Travel)
    - scope: Scope (e.g., Scope 3)
    - method: Estimation method (Spend-based, Fuel-based, Hybrid, Site-specific)
    - activity_data: Object with type, unit, estimated_quantity
    - emission_factor: Object with unit, value
    - emissions_kg: Estimated emissions in kg CO₂e
    - explanation: Explanation of the calculation
    """