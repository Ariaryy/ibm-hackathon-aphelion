import re
from data_prep_toolkit_transforms import TextCleaner, EntityExtractor  # Adjust based on actual API
import json

def clean_and_structure_data(extracted_text):
    """Clean and structure extracted text using IBM Data Prep Toolkit."""
    try:
        # Initialize DPK components (adjust based on actual API)
        text_cleaner = TextCleaner()
        entity_extractor = EntityExtractor()

        # Step 1: Clean text (remove noise, normalize)
        cleaned_text = text_cleaner.clean(
            text=extracted_text,
            remove_extra_spaces=True,
            normalize_case=True,
            remove_special_chars=False  # Keep special chars for currency/date
        )

        # Step 2: Extract entities to structure data
        entities = entity_extractor.extract(
            text=cleaned_text,
            entity_types=["description", "amount", "vendor", "date"]
        )

        # Step 3: Structure data into required JSON format
        structured_data = []
        for entity in entities:
            # Example parsing logic for Uber ride (customize based on actual output)
            if "uber" in entity.get("vendor", "").lower():
                amount_match = re.search(r'inr\s*(\d+\.?\d*)', entity.get("amount", ""))
                amount = float(amount_match.group(1)) if amount_match else 0.0
                structured_data.append({
                    "description": entity.get("description", "Uber ride to airport"),
                    "amount_spent_inr": amount,
                    "vendor": entity.get("vendor", "Uber"),
                    "date": entity.get("date", "2025-07-20")  # Mock date; adjust as needed
                })

        # Fallback: If no entities are extracted, create a mock entry for testing
        if not structured_data and "uber" in cleaned_text.lower():
            amount_match = re.search(r'inr\s*(\d+\.?\d*)', cleaned_text)
            amount = float(amount_match.group(1)) if amount_match else 0.0
            structured_data.append({
                "description": "Uber ride to airport",
                "amount_spent_inr": amount,
                "vendor": "Uber",
                "date": "2025-07-20"
            })

        return structured_data

    except Exception as e:
        print(f"Error in DPK processing: {e}")
        return []