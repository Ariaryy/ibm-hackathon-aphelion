import streamlit as st
import pandas as pd
import plotly.express as px
from aws_utils import upload_to_s3, extract_text_from_document
from dpk_cleaner import clean_and_structure_data
from emission_logic import estimate_emissions
from llm_reasoner import classify_and_estimate
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Aphelion - GHG Emissions Estimator", layout="wide")

def main():
    st.title("Aphelion: AI-Powered GHG Emissions Estimator")
    
    # Sidebar for inputs
    st.sidebar.header("Input Data")
    
    # Scope 1 and Scope 2 direct inputs
    scope1 = st.sidebar.number_input("Scope 1 Emissions (kg CO₂e)", min_value=0.0, value=0.0)
    scope2 = st.sidebar.number_input("Scope 2 Emissions (kg CO₂e)", min_value=0.0, value=0.0)
    
    # File upload for Scope 3
    uploaded_file = st.sidebar.file_uploader("Upload Invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uuid.uuid4()}.{uploaded_file.name.split('.')[-1]}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to S3
        s3_key = f"uploads/{uploaded_file.name}"
        bucket_name = os.getenv("AWS_S3_BUCKET")
        upload_to_s3(temp_file_path, bucket_name, s3_key)
        
        # Extract text using Textract
        extracted_text = extract_text_from_document(bucket_name, s3_key)
        
        # Clean and structure data using DPK
        structured_data = clean_and_structure_data(extracted_text)
        
        # Estimate emissions for each line item
        results = []
        for item in structured_data:
            llm_result = classify_and_estimate(item)
            emissions = estimate_emissions(llm_result)
            results.append({
                "Description": item["description"],
                "Vendor": item["vendor"],
                "Date": item["date"],
                "Amount (INR)": item["amount_spent_inr"],
                "Category": llm_result["category"],
                "Scope": llm_result["scope"],
                "Method": llm_result["method"],
                "Quantity": f"{llm_result['activity_data']['estimated_quantity']} {llm_result['activity_data']['unit']}",
                "Emission Factor": f"{llm_result['emission_factor']['value']} {llm_result['emission_factor']['unit']}",
                "Emissions (kg CO₂e)": llm_result["emissions_kg"],
                "Explanation": llm_result["explanation"]
            })
        
        # Calculate total Scope 3 emissions
        scope3_total = sum(result["Emissions (kg CO₂e)"] for result in results)
        
        # Display dashboard
        st.header("Emissions Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Emissions (kg CO₂e)")
            emissions_data = {
                "Scope": ["Scope 1", "Scope 2", "Scope 3"],
                "Emissions (kg CO₂e)": [scope1, scope2, scope3_total]
            }
            df_emissions = pd.DataFrame(emissions_data)
            fig = px.bar(df_emissions, x="Scope", y="Emissions (kg CO₂e)", title="Emissions by Scope")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Line-Item Emissions")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)
            
            # Download button for results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="emissions_results.csv",
                mime="text/csv"
            )
        
        # Clean up temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()