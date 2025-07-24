import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to S3."""
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False

def extract_text_from_document(bucket_name, s3_key):
    """Extract text from a document in S3 using AWS Textract."""
    textract_client = boto3.client(
        "textract",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    try:
        response = textract_client.analyze_document(
            Document={"S3Object": {"Bucket": bucket_name, "Name": s3_key}},
            FeatureTypes=["TABLES", "FORMS"]
        )
        # Simplified text extraction (process blocks as needed)
        extracted_text = ""
        for block in response["Blocks"]:
            if block["BlockType"] in ["LINE", "WORD"]:
                extracted_text += block.get("Text", "") + " "
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text with Textract: {e}")
        return ""