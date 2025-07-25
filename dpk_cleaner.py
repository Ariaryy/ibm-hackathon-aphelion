import os
import re
import json
import boto3
import PyPDF2
from pathlib import Path
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env_local (or .env) at the very beginning
load_dotenv(dotenv_path='.env.local') # Explicitly load .env_local

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURE THESE ===
# Retrieve values from environment variables, with fallbacks for safety
REGION = os.getenv("REGION", "ap-south-1") # Fallback for local testing if .env_local is missing
AGENT_ID = os.getenv("AGENT_ID", "24AZIE3K8M") # Fallback for local testing if .env_local is missing
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "KBMRPACFVM") # Fallback
SESSION_ID = "aphelion-invoice-processing-session" # Can be static or generated per run/session

# === SETUP BEDROCK CLIENT ===
def setup_bedrock_client():
    """Initialize Bedrock client with error handling."""
    try:
        # Client for Bedrock Agent Runtime - for invoking the agent
        client = boto3.client("bedrock-agent-runtime", region_name=REGION)
        logger.info(f"✅ Bedrock client initialized successfully for region: {REGION}")
        return client
    except Exception as e:
        logger.error(f"⚠️ Failed to configure Bedrock client: {e}")
        logger.info("Ensure AWS credentials are set up correctly. This can be done via:")
        logger.info("  1. AWS CLI configuration: `aws configure`")
        logger.info("  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
        logger.info("  3. IAM Role attached to an EC2 instance or ECS task (recommended for production)")
        return None

# === EXTRACT TEXT FROM PDF ===
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")

            text = "\n".join(text_parts).strip()
            if text:
                logger.info(f"Extracted text (first 100 chars): {text[:100]}...")
            else:
                logger.warning("No text extracted from PDF.")
            return text

    except Exception as e:
        logger.error(f"❌ Failed to extract text from PDF: {e}")
        return ""

# === DATA EXTRACTION ===
def extract_invoice_data(text: str) -> List[str]:
    """
    Extract invoice data as raw text strings using regular expressions.

    Args:
        text: Raw text from PDF

    Returns:
        List of raw text strings, one per invoice entry
    """
    # Regular expressions for extraction
    vendor_pattern = r"vendor:\s*([^\n:]+?)(?=\s*(?:date:|trip:|period:|amount:|\n|$))"
    date_pattern = r"date:\s*([^\n:]+?)(?=\s*(?:vendor:|trip:|period:|amount:|ride details:|electricity bill:|\n|$))"
    description_pattern = r"(?:trip|period):\s*([^\n:]+?)(?=\s*(?:vendor:|date:|amount:|\n|$))"
    amount_pattern = r"amount:\s*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)"

    # Split entries on double newlines
    entries = [entry.strip() for entry in text.split('\n\n') if entry.strip()]
    logger.info(f"Split into {len(entries)} potential entries")

    extracted_data = []

    for entry in entries:
        try:
            cleaned_entry = re.sub(r'\s+', ' ', entry.strip())

            vendor_match = re.search(vendor_pattern, cleaned_entry, re.IGNORECASE)
            date_match = re.search(date_pattern, cleaned_entry, re.IGNORECASE)
            description_match = re.search(description_pattern, cleaned_entry, re.IGNORECASE)
            amount_match = re.search(amount_pattern, cleaned_entry, re.IGNORECASE)

            vendor = vendor_match.group(1).strip().capitalize() if vendor_match else "Unknown"
            date = date_match.group(1).strip().capitalize() if date_match else "Unknown"
            description = description_match.group(1).strip() if description_match else "Service charge"
            amount = amount_match.group(1) if amount_match else "0.00"

            if vendor != "Unknown" or float(amount.replace(',', '')) > 0:
                raw_entry = f"Vendor: {vendor}\nDate: {date}\nDescription: {description}\nAmount: ₹{amount}"
                extracted_data.append(raw_entry)
                logger.info(f"Extracted raw entry: {raw_entry}")
            else:
                logger.warning(f"Skipping entry, no valid vendor or non-zero amount: '{entry[:50]}...'")

        except Exception as e:
            logger.warning(f"Failed to extract data from entry '{entry[:50]}...': {e}")

    return extracted_data

# === BEDROCK AGENT INVOCATION (YOUR CORE CONNECTION LOGIC) ===
def invoke_bedrock_agent(bedrock_client, raw_entry: str) -> Dict[str, Any]:
    """
    Sends a raw text entry to Bedrock agent and processes its streaming JSON response.

    Args:
        bedrock_client: Boto3 Bedrock client
        raw_entry: Raw text string to process (this will be the 'inputText')

    Returns:
        JSON response from Bedrock or error details
    """
    if not bedrock_client:
        logger.error("Bedrock client not available. Cannot invoke agent.")
        return {"raw_entry": raw_entry, "bedrock_error": "Bedrock client not initialized"}

    try:
        logger.info(f"Invoking Bedrock Agent with input: '{raw_entry[:100]}...'")
        
        # This is the direct invocation as shown in your example, but with correct parameters
        response = bedrock_client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=SESSION_ID,
            inputText=raw_entry,  # Your extracted raw data goes here
            # You can add other parameters if needed, e.g.:
            # enableTrace=True, # For detailed tracing in CloudWatch Logs
            # endSession=False, # Set to True if this is the last interaction in a session
            # promptCreationConfigurations={'excludePreviousThinkingSteps': True}, # If you don't want prior steps influencing the prompt
        )

        # The 'completion' from invoke_agent is a streaming response,
        # so we need to iterate through chunks to get the full result.
        completion = ""
        for chunk in response.get("completion", []):
            if "chunk" in chunk:
                completion += chunk["chunk"]["bytes"].decode("utf-8")

        if completion:
            try:
                # Assuming your Bedrock Agent is configured to return JSON
                bedrock_result = json.loads(completion)
                logger.info(f"Bedrock response (parsed JSON): {bedrock_result}")
                return bedrock_result
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from Bedrock. Raw response: {completion}")
                return {"raw_entry": raw_entry, "bedrock_error": "Invalid JSON response", "raw_response": completion}
        else:
            logger.error("No completion received from Bedrock agent.")
            return {"raw_entry": raw_entry, "bedrock_error": "No completion received"}

    except bedrock_client.exceptions.AccessDeniedException as e:
        logger.error(f"❌ Access Denied when calling Bedrock Agent. Check IAM permissions. Error: {e}")
        return {"raw_entry": raw_entry, "bedrock_error": f"Access Denied: {e}"}
    except bedrock_client.exceptions.ResourceNotFoundException as e:
        logger.error(f"❌ Bedrock Agent or Alias not found. Check AGENT_ID and AGENT_ALIAS_ID. Error: {e}")
        return {"raw_entry": raw_entry, "bedrock_error": f"Resource Not Found: {e}"}
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during agent invocation: {e}")
        return {"raw_entry": raw_entry, "bedrock_error": str(e)}

# === MAIN INVOICE PROCESSING PIPELINE ===
def process_invoices(pdf_path: str = "Vendor.pdf") -> List[Dict[str, Any]]:
    """
    Process invoices from PDF and send to Bedrock agent.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Bedrock-processed invoice data
    """
    bedrock_client = setup_bedrock_client()
    if not bedrock_client:
        logger.error("❌ Bedrock client setup failed. Cannot process invoices.")
        return [] # Exit if client setup failed

    try:
        logger.info("🔍 Processing input...")

        # Extract text from PDF
        logger.info(f"📄 Extracting text from PDF: {pdf_path}")
        input_text = extract_text_from_pdf(pdf_path)
        if not input_text:
            logger.error("❌ No text extracted from PDF. Exiting.")
            return []

        logger.info(f"📝 Extracted {len(input_text)} characters from PDF")

        # Extract structured data as raw text strings, one for each invoice entry
        logger.info("🔍 Extracting invoice data...")
        extracted_data = extract_invoice_data(input_text)
        if not extracted_data:
            logger.warning("❌ No invoice data extracted")
            return []

        logger.info(f"✅ Extracted {len(extracted_data)} invoice entries")

        # Send each raw invoice entry to Bedrock agent
        logger.info("🤖 Sending data to Bedrock agent...")
        processed_data = []
        for i, raw_entry in enumerate(extracted_data):
            logger.info(f"Processing item {i+1}/{len(extracted_data)}: {raw_entry[:50]}...")
            result = invoke_bedrock_agent(bedrock_client, raw_entry) # This is where the invocation happens
            processed_data.append(result)

        return processed_data

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return []

# === MAIN EXECUTION ===
def main():
    """Main execution function."""
    logger.info("🚀 Running invoice processing pipeline...\n")

    # Check if PDF exists
    pdf_path = "Vendor.pdf"
    if not Path(pdf_path).exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        logger.info("Please ensure 'Vendor.pdf' is in the same directory as this script, or provide the full path.")
        return

    # Process Vendor.pdf
    logger.info(f"🔍 Processing {pdf_path}...")
    results = process_invoices(pdf_path)

    if results:
        logger.info("\n✅ Final Structured Output:")
        print(json.dumps(results, indent=2, ensure_ascii=False)) # Pretty print the JSON output

        # --- Summary Statistics (unchanged, still useful) ---
        # total_amount = 0.0
        # for result in results:
        #     if 'raw_entry' in result:
        #         amount_match = re.search(r'Amount: ₹([\d,.]+)', result['raw_entry'])
        #         if amount_match:
        #             try:
        #                 total_amount += float(amount_match.group(1).replace(',', ''))
        #             except ValueError:
        #                 logger.warning(f"Could not parse amount from: {amount_match.group(1)}")

        # logger.info(f"\n📊 Summary:")
        # logger.info(f"Total invoices processed: {len(results)}")
        # logger.info(f"Total amount: ₹{total_amount:.2f}")

        # vendors = {}
        # for result in results:
        #     if 'raw_entry' in result:
        #         vendor_match = re.search(r'Vendor: ([^\n]+)', result['raw_entry'])
        #         vendor = vendor_match.group(1).strip() if vendor_match else 'Unknown'
        #         vendors[vendor] = vendors.get(vendor, 0) + 1

        # logger.info("Vendors found:")
        # for vendor, count in vendors.items():
        #     logger.info(f"  - {vendor}: {count} transactions")
    else:
        logger.warning("❌ No results generated from invoice processing.")

import pandas as pd

# Load the emission factor dataset
df = pd.read_csv("ghg-conversion-factors-2025-flat-format.csv")

import pandas as pd

def find_emission_factor(scope: str, activity_or_fuel: str, unit: str, df: pd.DataFrame):
    # Normalize
    if not scope.lower().startswith("scope"):
        scope = f"Scope {scope.strip()}"
    scope = scope.title()
    activity_or_fuel = activity_or_fuel.strip().title()
    unit = unit.strip().lower()

    # Match on Scope, Level 3 (fuel/activity), and Unit
    match = df[
        (df["Scope"].str.strip().str.title() == scope) &
        (df["Level 3"].str.strip().str.title().str.contains(activity_or_fuel, na=False)) &
        (df["UOM"].str.strip().str.lower() == unit)
    ]

    if match.empty:
        return None

    row = match.iloc[0]
    return {
        "emission_factor_id": row["ID"],
        "emission_factor": float(row["GHG Conversion Factor 2025"]),
        "unit": row["UOM"],
        "ghg_unit": row["GHG/Unit"]
    }

def apply_emission_factors(agent_output: list, df: pd.DataFrame):
    results = []

    for item in agent_output:
        scope = item.get("scope")
        activity = item.get("activity")
        unit = item.get("unit")
        quantity = item.get("quantity")

        factor_data = find_emission_factor(scope, activity, unit, df)

        if not factor_data:
            results.append({
                "input": item,
                "status": "No emission factor found",
                "co2e_estimate_kg": None
            })
            continue

        co2e = quantity * factor_data["emission_factor"]

        results.append({
            "input": item,
            "scope": scope,
            "category": activity,
            "activity_quantity": quantity,
            "unit": unit,
            "emission_factor_used": factor_data,
            "co2e_estimate_kg": round(co2e, 4),
            "status": "Success"
        })

    return results


def testMain():
    bedrock_client = setup_bedrock_client()
    if not bedrock_client:
        logger.error("❌ Bedrock client setup failed. Cannot process invoices.")
        return [] # Exit if client setup failed
    

    prompt = """
Vendor: Uber India
Date: July 24, 2025

Ride Details:
- Trip: Office to Airport
- Kilmeters: 5
- Vehicle: Car

Vendor: Tata Power
Date: July 1, 2025

Electricity Bill:
- Period: June 2025
- Units Consumed: 600 kWh
- Amount: ₹4800
"""

    result = invoke_bedrock_agent(bedrock_client, prompt)

    # result_json = (json.dumps(result, indent=2, ensure_ascii=False))
    factor = apply_emission_factors(result, df)

    print(json.dumps(factor, indent=2))

if __name__ == "__main__":
    testMain()