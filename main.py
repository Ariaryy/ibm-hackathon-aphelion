import gradio as gr
import pandas as pd
import boto3
import PyPDF2
import json
import os
import logging
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

from extended_carbon_app import GHGFactorMatcher, create_emissions_charts

# --- INITIAL SETUP ---
load_dotenv(dotenv_path=".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & GLOBAL VARIABLES ---
REGION = os.getenv("REGION", "ap-south-1")
FLOW_ID = os.getenv("FLOW_ID")  # Primary flow for Scope 1&2
FLOW_ALIAS_ID = os.getenv("FLOW_ALIAS_ID")
SCOPE3_FLOW_ID = os.getenv("SCOPE3_FLOW_ID")  # New: Scope 3 flow ID
SCOPE3_FLOW_ALIAS_ID = os.getenv("SCOPE3_FLOW_ALIAS_ID")  # New: Scope 3 flow alias
AGENT_ID = os.getenv("AGENT_ID")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID")

# Validate environment variables
if not FLOW_ID or not FLOW_ALIAS_ID:
    logger.error(
        "FATAL: FLOW_ID and FLOW_ALIAS_ID must be set in your .env.local file."
    )

if not SCOPE3_FLOW_ID or not SCOPE3_FLOW_ALIAS_ID:
    logger.warning(
        "WARNING: SCOPE3_FLOW_ID and SCOPE3_FLOW_ALIAS_ID not set. Scope 3 analysis disabled."
    )

if not AGENT_ID or not AGENT_ALIAS_ID:
    logger.error(
        "FATAL: AGENT_ID and AGENT_ALIAS_ID must be set in your .env.local file."
    )

# --- GLOBAL CLIENTS AND DATA ---
BEDROCK_RUNTIME_CLIENT = None
BEDROCK_AGENT_CLIENT = None
GHG_FACTORS_DF = None
CURRENT_RESULTS = []
CURRENT_SCOPE3_RESULTS = []  # New: Store Scope 3 results separately
CURRENT_SESSION_ID = None


def initialize_globals():
    """Initialize global resources like the Boto3 clients and GHG data."""
    global BEDROCK_RUNTIME_CLIENT, BEDROCK_AGENT_CLIENT, GHG_FACTORS_DF

    # Setup Bedrock clients
    try:
        BEDROCK_RUNTIME_CLIENT = boto3.client(
            "bedrock-agent-runtime", region_name=REGION
        )
        BEDROCK_AGENT_CLIENT = boto3.client("bedrock-agent-runtime", region_name=REGION)
        logger.info(f"✅ Bedrock clients initialized successfully for region: {REGION}")
    except Exception as e:
        logger.error(f"⚠️ Failed to configure Bedrock clients: {e}")

    # Load GHG conversion factors
    try:
        GHG_FACTORS_DF = pd.read_csv("ghg-conversion-factors-2025-flat-format.csv")
        logger.info(f"✅ Loaded {len(GHG_FACTORS_DF)} GHG conversion factors.")
    except FileNotFoundError:
        logger.error("FATAL: 'ghg-conversion-factors-2025-flat-format.csv' not found.")
    except Exception as e:
        logger.error(f"⚠️ Failed to load GHG conversion factors: {e}")


# --- EXISTING FUNCTIONS (keeping extract_text_from_pdf and other utilities) ---


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        logger.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"❌ Failed to extract text from PDF: {e}")
        raise


# --- BEDROCK FLOW FUNCTIONS ---


def clean_json_response(text):
    # Remove ```json from the beginning and ``` from the end
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # Remove '```json'
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # Remove '```'
    return cleaned.strip()


def invoke_bedrock_flow(
    text: str, flow_type: str = "primary"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Send text to Bedrock Flow and get structured JSON response.

    Args:
        text: Input text to process
        flow_type: "primary" for Scope 1&2, "scope3" for Scope 3
    """
    if not BEDROCK_RUNTIME_CLIENT:
        raise ConnectionError("Bedrock client not available. Cannot invoke flow.")

    # Select appropriate flow based on type
    if flow_type == "scope3":
        if not SCOPE3_FLOW_ID or not SCOPE3_FLOW_ALIAS_ID:
            raise ValueError(
                "Scope 3 flow not configured. Please set SCOPE3_FLOW_ID and SCOPE3_FLOW_ALIAS_ID."
            )
        flow_id = SCOPE3_FLOW_ID
        flow_alias_id = SCOPE3_FLOW_ALIAS_ID
        logger.info(
            f"🚀 Invoking Scope 3 Flow with input length: {len(text)} characters"
        )
    else:
        flow_id = FLOW_ID
        flow_alias_id = FLOW_ALIAS_ID
        logger.info(
            f"🚀 Invoking Primary Flow with input length: {len(text)} characters"
        )

    if not text or not text.strip():
        raise ValueError("Input text is empty or None")

    try:
        response = BEDROCK_RUNTIME_CLIENT.invoke_flow(
            flowIdentifier=flow_id,
            flowAliasIdentifier=flow_alias_id,
            inputs=[
                {
                    "nodeOutputName": "document",
                    "nodeName": "FlowInputNode",
                    "content": {"document": text},
                }
            ],
        )

        logger.info("✅ Flow invocation successful, processing response stream...")

        final_response = None
        event_count = 0

        for event in response.get("responseStream", []):
            event_count += 1
            logger.debug(f"Processing event #{event_count}: {list(event.keys())}")

            if "flowOutputEvent" in event:
                final_response = event
                logger.info("📦 Found flowOutputEvent in stream")
                break
            elif "flowCompletionEvent" in event:
                logger.info("🏁 Flow completion event received")
            elif "internalServerException" in event:
                error_msg = event["internalServerException"].get(
                    "message", "Unknown server error"
                )
                logger.error(f"❌ Internal server exception: {error_msg}")
                raise Exception(f"Bedrock internal server error: {error_msg}")
            elif "validationException" in event:
                error_msg = event["validationException"].get(
                    "message", "Validation failed"
                )
                logger.error(f"❌ Validation exception: {error_msg}")
                raise ValueError(f"Flow validation error: {error_msg}")
            elif "throttlingException" in event:
                error_msg = event["throttlingException"].get(
                    "message", "Request throttled"
                )
                logger.error(f"⏱️ Throttling exception: {error_msg}")
                raise Exception(f"Bedrock throttling error: {error_msg}")

        if final_response is None:
            logger.error("❌ No flowOutputEvent found in response stream")
            raise ValueError(
                "Flow did not produce a valid flowOutputEvent. Check your flow configuration."
            )

        try:
            content = final_response["flowOutputEvent"]["content"]
            logger.debug(f"Flow output content keys: {list(content.keys())}")

            output_data = None
            raw_json_str = ""

            possible_output_keys = ["FlowOutputNode", "output", "result", "document"]

            for key in possible_output_keys:
                if key in content:
                    output_data = content[key]
                    logger.info(f"✅ Found output data under key: '{key}'")
                    break

            if output_data is None:
                if content:
                    first_key = list(content.keys())[0]
                    output_data = content[first_key]
                    logger.warning(
                        f"⚠️ Using first available content key: '{first_key}'"
                    )
                else:
                    raise ValueError("No content found in flowOutputEvent")

            if isinstance(output_data, dict) and "document" in output_data:
                raw_json_str = output_data["document"].get("value", "")
            elif isinstance(output_data, dict) and "value" in output_data:
                raw_json_str = output_data["value"]
            elif isinstance(output_data, str):
                raw_json_str = output_data
            else:
                raw_json_str = str(output_data)

            logger.info(f"📝 Raw JSON string length: {len(raw_json_str)} characters")

            if not raw_json_str.strip():
                raise ValueError("Empty response from Bedrock Flow")

            try:
                parsed_result = json.loads(clean_json_response(raw_json_str))
                logger.info(
                    f"✅ Successfully parsed JSON with {len(parsed_result) if isinstance(parsed_result, list) else 1} items"
                )

                if isinstance(parsed_result, list):
                    return parsed_result, raw_json_str
                else:
                    return [parsed_result], raw_json_str

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                logger.error(f"Raw response preview: {raw_json_str[:200]}...")
                raise ValueError(
                    f"Could not parse JSON response from Bedrock Flow: {e}"
                )

        except KeyError as e:
            logger.error(f"❌ Unexpected response structure: missing key {e}")
            logger.error(
                f"Available keys: {list(final_response.get('flowOutputEvent', {}).get('content', {}).keys())}"
            )
            raise ValueError(f"Unexpected response structure from Bedrock Flow: {e}")

    except Exception as e:
        if isinstance(e, (ConnectionError, ValueError)):
            raise
        else:
            logger.error(
                f"❌ Unexpected error during flow invocation: {e}", exc_info=True
            )
            raise Exception(f"Flow invocation failed: {str(e)}")


# --- SCOPE 3 PROCESSING FUNCTIONS ---


def process_scope3_data(scope3_results: List[Dict]) -> List[Dict]:
    """Process Scope 3 data which already contains estimated emissions."""
    processed_results = []

    for item in scope3_results:
        # Scope 3 data already has estimated emissions, so we just standardize the format
        processed_item = {
            "activity": f"{item.get('category', 'Unknown')} - {item.get('subcategory', 'Unknown')}",
            "category": item.get("category", "Unknown"),
            "subcategory": item.get("subcategory", "Unknown"),
            "scope": "Scope 3",
            "amount_inr": item.get("amountINR", 0),
            "calculated_emissions": item.get("estimatedEmissionsKg", 0),
            "unit": "INR",
            "emission_unit": "kg CO₂e",
            "data_source": "Scope 3 Flow Analysis",
            "match_confidence": 1.0,  # High confidence since it's directly from the flow
        }
        processed_results.append(processed_item)

        logger.info(
            f"✅ Processed Scope 3 item: {processed_item['activity']} - {processed_item['calculated_emissions']} kg CO₂e"
        )

    return processed_results


def create_scope3_visualizations(
    scope3_results: List[Dict],
) -> Tuple[go.Figure, go.Figure]:
    """Create specific visualizations for Scope 3 data."""
    if not scope3_results:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No Scope 3 data available", x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, empty_fig

    # 1. Emissions by Category
    category_emissions = {}
    for item in scope3_results:
        category = item.get("category", "Unknown")
        emissions = item.get("calculated_emissions", 0)
        category_emissions[category] = category_emissions.get(category, 0) + emissions

    category_fig = go.Figure(
        data=[
            go.Bar(
                x=list(category_emissions.keys()),
                y=list(category_emissions.values()),
                marker_color="#9B59B6",
                text=[f"{v:.2f} kg CO₂e" for v in category_emissions.values()],
                textposition="auto",
            )
        ]
    )

    total_scope3_emissions = sum(category_emissions.values())
    category_fig.update_layout(
        title=f"Scope 3 Emissions by Category - Total: {total_scope3_emissions:.2f} kg CO₂e",
        xaxis_title="Categories",
        yaxis_title="Emissions (kg CO₂e)",
        xaxis_tickangle=-45,
        height=400,
    )

    # 2. Detailed breakdown by subcategory
    subcategory_data = []
    for item in scope3_results:
        subcategory_data.append(
            {
                "category": item.get("category", "Unknown"),
                "subcategory": item.get("subcategory", "Unknown"),
                "emissions": item.get("calculated_emissions", 0),
                "amount": item.get("amount_inr", 0),
            }
        )

    # Sort by emissions
    subcategory_data.sort(key=lambda x: x["emissions"], reverse=True)
    top_subcategories = subcategory_data[:10]  # Top 10

    subcategory_fig = go.Figure(
        data=[
            go.Bar(
                y=[
                    f"{item['category'][:20]}<br>{item['subcategory'][:20]}"
                    for item in top_subcategories
                ],
                x=[item["emissions"] for item in top_subcategories],
                orientation="h",
                marker_color="#8E44AD",
                text=[f"{item['emissions']:.2f}" for item in top_subcategories],
                textposition="auto",
            )
        ]
    )

    subcategory_fig.update_layout(
        title="Top 10 Scope 3 Subcategories by Emissions",
        xaxis_title="Emissions (kg CO₂e)",
        yaxis_title="Category - Subcategory",
        height=500,
        margin=dict(l=200),
    )

    return category_fig, subcategory_fig


# --- COMBINED VISUALIZATION FUNCTIONS ---


def create_combined_emissions_charts(
    scope12_results: List[Dict], scope3_results: List[Dict]
) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Create combined visualizations showing all scopes together."""

    # 1. Combined Scope Overview
    scope_totals = {"Scope 1": 0, "Scope 2": 0, "Scope 3": 0}

    # Add Scope 1&2 emissions
    for res in scope12_results:
        scope_key = res.get("scope", "").strip()
        if not scope_key.startswith("Scope"):
            scope_key = f"Scope {scope_key}"
        emissions = float(res.get("calculated_emissions", 0))
        if scope_key in scope_totals:
            scope_totals[scope_key] += emissions

    # Add Scope 3 emissions
    scope3_total = sum(res.get("calculated_emissions", 0) for res in scope3_results)
    scope_totals["Scope 3"] += scope3_total

    # Filter out zero values
    filtered_scope_data = {k: v for k, v in scope_totals.items() if v > 0}
    colors = ["#E74C3C", "#3498DB", "#9B59B6"]  # Red, Blue, Purple

    combined_scope_fig = go.Figure(
        data=[
            go.Bar(
                x=list(filtered_scope_data.keys()),
                y=list(filtered_scope_data.values()),
                marker_color=colors[: len(filtered_scope_data)],
                text=[f"{v:,.2f} kg CO₂e" for v in filtered_scope_data.values()],
                textposition="auto",
            )
        ]
    )

    total_emissions = sum(filtered_scope_data.values())
    combined_scope_fig.update_layout(
        title=f"Complete Emissions Overview - Total: {total_emissions:,.2f} kg CO₂e",
        xaxis_title="Emission Scopes",
        yaxis_title="Emissions (kg CO₂e)",
        height=400,
    )

    # 2. Combined Pie Chart
    combined_pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=list(filtered_scope_data.keys()),
                values=list(filtered_scope_data.values()),
                marker_colors=colors[: len(filtered_scope_data)],
                textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b><br>"
                + "Emissions: %{value:.2f} kg CO₂e<br>"
                + "Percentage: %{percent}<br>"
                + "<extra></extra>",
            )
        ]
    )

    combined_pie_fig.update_layout(
        title="Complete Carbon Footprint Distribution", height=400
    )

    # 3. Detailed Breakdown - Top activities from all scopes
    all_activities = []

    # Add Scope 1&2 activities
    for res in scope12_results:
        if res.get("calculated_emissions", 0) > 0:
            all_activities.append(
                {
                    "activity": res.get("activity", "Unknown"),
                    "scope": res.get("scope", "Unknown"),
                    "emissions": res.get("calculated_emissions", 0),
                    "type": "Direct/Indirect",
                }
            )

    # Add Scope 3 activities
    for res in scope3_results:
        if res.get("calculated_emissions", 0) > 0:
            all_activities.append(
                {
                    "activity": res.get("activity", "Unknown"),
                    "scope": "Scope 3",
                    "emissions": res.get("calculated_emissions", 0),
                    "type": "Value Chain",
                }
            )

    # Sort and take top 15
    all_activities.sort(key=lambda x: x["emissions"], reverse=True)
    top_activities = all_activities[:15]

    # Color by scope
    colors_map = {"Scope 1": "#E74C3C", "Scope 2": "#3498DB", "Scope 3": "#9B59B6"}
    bar_colors = [colors_map.get(act["scope"], "#95A5A6") for act in top_activities]

    detailed_fig = go.Figure(
        data=[
            go.Bar(
                y=[
                    act["activity"][:40] + "..."
                    if len(act["activity"]) > 40
                    else act["activity"]
                    for act in top_activities
                ],
                x=[act["emissions"] for act in top_activities],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{act['emissions']:.2f}" for act in top_activities],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>"
                + "Emissions: %{x:.2f} kg CO₂e<br>"
                + "Scope: %{customdata}<br>"
                + "<extra></extra>",
                customdata=[act["scope"] for act in top_activities],
            )
        ]
    )

    detailed_fig.update_layout(
        title="Top 15 Activities Across All Scopes",
        xaxis_title="Emissions (kg CO₂e)",
        yaxis_title="Activities",
        height=600,
        margin=dict(l=250),
    )

    return combined_scope_fig, combined_pie_fig, detailed_fig


# --- ANALYSIS FUNCTIONS ---


def run_scope12_analysis(pdf_file):
    """Run analysis for Scope 1&2 data using primary flow."""
    global CURRENT_RESULTS

    if not all([BEDROCK_RUNTIME_CLIENT, GHG_FACTORS_DF is not None]):
        return "## ⚠️ Error: Primary flow not configured", None, None, None, None, None

    try:
        logger.info(f"🚀 Starting Scope 1&2 analysis for PDF: {pdf_file.name}")

        # Extract text and process with primary flow
        input_text = extract_text_from_pdf(pdf_file.name)
        rag_results, raw_json_str = invoke_bedrock_flow(input_text, "primary")

        # Match to GHG factors (using existing matcher)
        matcher = GHGFactorMatcher(GHG_FACTORS_DF)
        final_results = matcher.match_activity_to_ghg_factor(rag_results)

        CURRENT_RESULTS = final_results

        # Create visualizations (using existing function)
        scope_chart, activities_chart, pie_chart, category_chart = (
            create_emissions_charts(final_results)
        )

        # Create results DataFrame
        results_df = pd.DataFrame(final_results)

        total_emissions = sum(r.get("calculated_emissions", 0) for r in final_results)
        matched_count = len(
            [r for r in final_results if r.get("calculated_emissions", 0) > 0]
        )

        summary = f"""
## 📊 Scope 1&2 Analysis Results

**Summary:**
- Total Activities: {len(final_results)}
- Successfully Matched: {matched_count}
- Total Emissions: {total_emissions:,.2f} kg CO₂e
"""

        return (
            summary,
            scope_chart,
            activities_chart,
            pie_chart,
            category_chart,
            results_df,
        )

    except Exception as e:
        logger.error(f"Scope 1&2 analysis failed: {e}")
        return f"## ❌ Error: {str(e)}", None, None, None, None, None


def run_scope3_analysis(pdf_file):
    """Run analysis for Scope 3 data using scope 3 flow."""
    global CURRENT_SCOPE3_RESULTS

    if not SCOPE3_FLOW_ID or not SCOPE3_FLOW_ALIAS_ID:
        return "## ⚠️ Error: Scope 3 flow not configured", None, None

    try:
        logger.info(f"🚀 Starting Scope 3 analysis for PDF: {pdf_file.name}")

        # Extract text and process with Scope 3 flow
        input_text = extract_text_from_pdf(pdf_file.name)
        scope3_raw_results, raw_json_str = invoke_bedrock_flow(input_text, "scope3")

        # Process Scope 3 data (no GHG matching needed as emissions are pre-calculated)
        final_results = process_scope3_data(scope3_raw_results)

        CURRENT_SCOPE3_RESULTS = final_results

        # Create Scope 3 specific visualizations
        category_chart, subcategory_chart = create_scope3_visualizations(final_results)

        total_emissions = sum(r.get("calculated_emissions", 0) for r in final_results)

        summary = f"""
## 🔗 Scope 3 Analysis Results

**Summary:**
- Total Categories: {len(set(r.get("category") for r in final_results))}
- Total Activities: {len(final_results)}
- Total Emissions: {total_emissions:.2f} kg CO₂e
- Total Spend: ₹{sum(r.get("amount_inr", 0) for r in final_results):,.2f}
"""

        return summary, category_chart, subcategory_chart

    except Exception as e:
        logger.error(f"Scope 3 analysis failed: {e}")
        return f"## ❌ Error: {str(e)}", None, None


def run_combined_analysis():
    """Create combined analysis of both Scope 1&2 and Scope 3 data."""
    global CURRENT_RESULTS, CURRENT_SCOPE3_RESULTS

    if not CURRENT_RESULTS and not CURRENT_SCOPE3_RESULTS:
        return (
            "## ⚠️ No data available. Please run individual analyses first.",
            None,
            None,
            None,
        )

    try:
        # Create combined visualizations
        combined_scope_chart, combined_pie_chart, detailed_breakdown = (
            create_combined_emissions_charts(CURRENT_RESULTS, CURRENT_SCOPE3_RESULTS)
        )

        # Calculate totals
        scope12_total = sum(r.get("calculated_emissions", 0) for r in CURRENT_RESULTS)
        scope3_total = sum(
            r.get("calculated_emissions", 0) for r in CURRENT_SCOPE3_RESULTS
        )
        grand_total = scope12_total + scope3_total

        summary = f"""
## 🌍 Complete Carbon Footprint Analysis

**Overall Summary:**
- Scope 1&2 Emissions: {scope12_total:,.2f} kg CO₂e ({(scope12_total / grand_total * 100):.1f}%)
- Scope 3 Emissions: {scope3_total:.2f} kg CO₂e ({(scope3_total / grand_total * 100):.1f}%)
- **Total Carbon Footprint: {grand_total:,.2f} kg CO₂e ({grand_total / 1000:.2f} tonnes CO₂e)**

**Key Insights:**
- Scope 1&2 Activities: {len(CURRENT_RESULTS)}
- Scope 3 Activities: {len(CURRENT_SCOPE3_RESULTS)}
- Total Activities Analyzed: {len(CURRENT_RESULTS) + len(CURRENT_SCOPE3_RESULTS)}
"""

        return summary, combined_scope_chart, combined_pie_chart, detailed_breakdown

    except Exception as e:
        logger.error(f"Combined analysis failed: {e}")
        return f"## ❌ Error: {str(e)}", None, None, None


# --- ENHANCED GRADIO UI ---


def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(), title="AI Carbon Footprint Analyzer"
    ) as demo:
        gr.Markdown("# 🌍 AI-Powered Carbon Footprint Analyzer")
        gr.Markdown(
            "Comprehensive analysis supporting both direct emissions (Scope 1&2) and value chain emissions (Scope 3)"
        )

        # --- FILE UPLOAD SECTION ---
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📄 Document Upload")
                scope12_pdf = gr.File(
                    label="Scope 1&2 Report (Energy, Fuel, etc.)", file_types=[".pdf"]
                )
                scope3_pdf = gr.File(
                    label="Scope 3 Report (Business Travel, Procurement, etc.)",
                    file_types=[".pdf"],
                )

                with gr.Row():
                    analyze_scope12_btn = gr.Button(
                        "🔥 Analyze Scope 1&2", variant="primary"
                    )
                    analyze_scope3_btn = gr.Button(
                        "🔗 Analyze Scope 3", variant="primary"
                    )
                    combined_analysis_btn = gr.Button(
                        "🌍 Combined Analysis", variant="secondary"
                    )

        # --- RESULTS TABS ---
        with gr.Tabs():
            # Tab 1: Scope 1&2 Results
            with gr.TabItem("🔥 Scope 1&2 Results"):
                scope12_summary = gr.Markdown(
                    "Upload a Scope 1&2 report to see direct and indirect emissions analysis."
                )

                with gr.Row():
                    scope12_scope_chart = gr.Plot(label="Scope Overview")
                    scope12_pie_chart = gr.Plot(label="Emissions Distribution")

                with gr.Row():
                    scope12_activities_chart = gr.Plot(label="Top Activities")
                    scope12_category_chart = gr.Plot(label="By Category")

                scope12_table = gr.DataFrame(label="Detailed Scope 1&2 Results")

            # Tab 2: Scope 3 Results
            with gr.TabItem("🔗 Scope 3 Results"):
                scope3_summary = gr.Markdown(
                    "Upload a Scope 3 report to see value chain emissions analysis."
                )

                with gr.Row():
                    scope3_category_chart = gr.Plot(label="Emissions by Category")
                    scope3_subcategory_chart = gr.Plot(label="Top Subcategories")

            # Tab 3: Combined Analysis
            with gr.TabItem("🌍 Complete Footprint"):
                combined_summary = gr.Markdown(
                    "Run both individual analyses first to see your complete carbon footprint."
                )

                with gr.Row():
                    combined_scope_chart = gr.Plot(label="Complete Scope Overview")
                    combined_pie_chart = gr.Plot(label="Total Distribution")

                combined_detailed_chart = gr.Plot(
                    label="Top Activities Across All Scopes"
                )

            # Tab 4: AI Assistant (existing chatbot functionality)
            with gr.TabItem("💬 AI Assistant"):
                gr.Markdown("## 🤖 Chat with AI Carbon Expert")

                gr.Markdown("## 💡 AI-Powered Recommendations")
                suggestions_button = gr.Button(
                    "🚀 Analyze Report", variant="primary", size="lg"
                )
                suggestions_output = gr.Markdown(
                    value="Click the button above to see personalized suggestions for reducing your carbon footprint."
                )

                chatbot = gr.Chatbot(
                    value=[], label="AI Carbon Footprint Expert", show_label=True
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask about your carbon footprint",
                        placeholder="e.g., Compare my Scope 1 vs Scope 3 emissions, What are my top reduction opportunities?",
                        scale=4,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

        # --- EVENT HANDLERS ---

        # Scope 1&2 Analysis
        analyze_scope12_btn.click(
            fn=run_scope12_analysis,
            inputs=[scope12_pdf],
            outputs=[
                scope12_summary,
                scope12_scope_chart,
                scope12_activities_chart,
                scope12_pie_chart,
                scope12_category_chart,
                scope12_table,
            ],
        )

        # Scope 3 Analysis
        analyze_scope3_btn.click(
            fn=run_scope3_analysis,
            inputs=[scope3_pdf],
            outputs=[scope3_summary, scope3_category_chart, scope3_subcategory_chart],
        )

        # Combined Analysis
        combined_analysis_btn.click(
            fn=run_combined_analysis,
            outputs=[
                combined_summary,
                combined_scope_chart,
                combined_pie_chart,
                combined_detailed_chart,
            ],
        )

        # Chatbot functionality (enhanced for multi-scope)
        def enhanced_chatbot_with_agent(
            message: str, history: List[List[str]]
        ) -> List[List[str]]:
            """Enhanced chatbot with both Scope 1&2 and Scope 3 context."""
            global CURRENT_RESULTS, CURRENT_SCOPE3_RESULTS, CURRENT_SESSION_ID

            if not message.strip():
                return history

            try:
                # Prepare comprehensive context
                context_prompt = ""

                if CURRENT_RESULTS or CURRENT_SCOPE3_RESULTS:
                    scope12_total = sum(
                        res.get("calculated_emissions", 0) for res in CURRENT_RESULTS
                    )
                    scope3_total = sum(
                        res.get("calculated_emissions", 0)
                        for res in CURRENT_SCOPE3_RESULTS
                    )
                    grand_total = scope12_total + scope3_total

                    context_prompt = f"""
CONTEXT: The user has analyzed their complete carbon footprint with the following results:

SCOPE 1&2 DATA:
- Total Scope 1&2 emissions: {scope12_total:,.2f} kg CO₂e
- Number of activities: {len(CURRENT_RESULTS)}
- Main sources: {", ".join([r.get("activity", "Unknown")[:30] for r in sorted(CURRENT_RESULTS, key=lambda x: x.get("calculated_emissions", 0), reverse=True)[:3]])}

SCOPE 3 DATA:
- Total Scope 3 emissions: {scope3_total:.2f} kg CO₂e
- Number of activities: {len(CURRENT_SCOPE3_RESULTS)}
- Main categories: {", ".join(list(set([r.get("category", "Unknown") for r in CURRENT_SCOPE3_RESULTS])))}

OVERALL:
- Total carbon footprint: {grand_total:,.2f} kg CO₂e ({grand_total / 1000:.2f} tonnes)
- Scope 1&2 percentage: {(scope12_total / grand_total * 100):.1f}%
- Scope 3 percentage: {(scope3_total / grand_total * 100):.1f}%

User's question: {message}

Please provide a comprehensive response considering their complete carbon footprint data.
"""
                else:
                    context_prompt = f"""
The user hasn't uploaded any emissions data yet for either Scope 1&2 or Scope 3.

User's question: {message}

Please provide general guidance about comprehensive carbon footprint analysis and encourage them to upload both types of reports.
"""

                # Get response from agent (reusing existing agent function)
                agent_response, session_id = invoke_bedrock_agent(
                    context_prompt, CURRENT_SESSION_ID
                )
                CURRENT_SESSION_ID = session_id

                history.append([message, agent_response])
                return history

            except Exception as e:
                logger.error(f"Enhanced chatbot error: {e}")
                error_message = (
                    "I'm having trouble processing your request. Please try again."
                )
                history.append([message, error_message])
                return history

        def send_message(message, history):
            if message.strip():
                updated_history = enhanced_chatbot_with_agent(message, history)
                return "", updated_history
            return message, history

        def clear_chat():
            global CURRENT_SESSION_ID
            CURRENT_SESSION_ID = None
            return []

        send_button.click(
            fn=send_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )

        msg_input.submit(
            fn=send_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )

        prompt_state = gr.State(
            "Provide with comprehensive suggestions for the above prompt."
        )
        chat_history_state = gr.State([[""]])  # Or the real chat history

        def auto_suggest(_ignored, history):
            prompt = "Provide with comprehensive suggestions for the above prompt."
            updated_history = enhanced_chatbot_with_agent(prompt, history)
            return updated_history[-1][1]  # Just the agent's reply string

        dummy = gr.State()

        suggestions_button.click(
            fn=auto_suggest,
            inputs=[dummy, chat_history_state],
            outputs=[suggestions_output],
        )

        # Add clear chat button
        with gr.Row():
            clear_chat_button = gr.Button("🗑️ Clear Chat", variant="secondary")
            clear_chat_button.click(fn=clear_chat, outputs=[chatbot])

        # --- HELP SECTION ---
        gr.Markdown("""
        ## 🚀 Getting Started
        
        ### For Complete Carbon Footprint Analysis:
        
        1. **Scope 1&2 Analysis**: Upload reports containing direct emissions data (fuel consumption, electricity usage, etc.)
        2. **Scope 3 Analysis**: Upload reports containing value chain data (business travel, procurement, employee commuting, etc.)
        3. **Combined Analysis**: After running both analyses, get a complete overview of your carbon footprint
        4. **AI Assistant**: Ask questions about your emissions data and get personalized recommendations
        
        ### Document Types:
        
        **Scope 1&2 Reports Should Contain:**
        - Fuel consumption (diesel, petrol, natural gas, etc.)
        - Electricity usage by location/grid
        - Direct process emissions
        - On-site energy generation
        
        **Scope 3 Reports Should Contain:**
        - Business travel expenses and categories
        - Employee commuting data
        - Procurement and supplier information
        - Waste and water treatment
        - Transportation and logistics costs
        
        ### Example Questions for AI Assistant:
        - "What percentage of my emissions come from Scope 3?"
        - "Which scope should I prioritize for reduction?"
        - "Compare my business travel vs fuel consumption emissions"
        - "What are the top 5 emission sources across all scopes?"
        - "How can I reduce my value chain emissions?"
        """)

    return demo


# --- AGENT FUNCTIONS (Enhanced for multi-scope) ---


def invoke_bedrock_agent(prompt: str, session_id: str = None) -> Tuple[str, str]:
    """
    Invoke Bedrock Agent for suggestions and chatbot responses.
    [Keep existing implementation but enhance context handling]
    """
    if not BEDROCK_AGENT_CLIENT:
        raise ConnectionError("Bedrock Agent client not available.")

    try:
        # Generate session ID if not provided
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        logger.info(f"🤖 Invoking Bedrock Agent with session: {session_id[:8]}...")

        response = BEDROCK_AGENT_CLIENT.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=prompt,
        )

        # Process streaming response
        agent_response = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    chunk_text = chunk["bytes"].decode("utf-8")
                    agent_response += chunk_text

        logger.info(f"✅ Agent response length: {len(agent_response)} characters")
        return agent_response.strip(), session_id

    except Exception as e:
        logger.error(f"❌ Error invoking Bedrock Agent: {e}", exc_info=True)
        raise Exception(f"Agent invocation failed: {str(e)}")


def generate_suggestions_with_agent(
    scope12_results: List[Dict[str, Any]], scope3_results: List[Dict[str, Any]] = None
) -> str:
    """Generate comprehensive AI-powered suggestions using both scope data."""
    try:
        # Calculate totals
        scope12_total = sum(
            res.get("calculated_emissions", 0) for res in scope12_results
        )
        scope3_total = sum(
            res.get("calculated_emissions", 0) for res in scope3_results or []
        )
        grand_total = scope12_total + scope3_total

        # Prepare comprehensive data for agent
        emissions_summary = {
            "scope_12_emissions_kg_co2e": scope12_total,
            "scope_3_emissions_kg_co2e": scope3_total,
            "total_emissions_kg_co2e": grand_total,
            "total_emissions_tonnes_co2e": grand_total / 1000,
            "scope_breakdown": {
                "scope_12_percentage": (scope12_total / grand_total * 100)
                if grand_total > 0
                else 0,
                "scope_3_percentage": (scope3_total / grand_total * 100)
                if grand_total > 0
                else 0,
            },
            "top_scope12_activities": [
                {
                    "activity": act.get("activity", "Unknown"),
                    "emissions_kg_co2e": act.get("calculated_emissions", 0),
                    "scope": act.get("scope", "Unknown"),
                }
                for act in sorted(
                    scope12_results,
                    key=lambda x: x.get("calculated_emissions", 0),
                    reverse=True,
                )[:5]
            ],
            "top_scope3_categories": [
                {
                    "category": cat,
                    "emissions_kg_co2e": sum(
                        r.get("calculated_emissions", 0)
                        for r in scope3_results or []
                        if r.get("category") == cat
                    ),
                }
                for cat in list(
                    set(r.get("category", "Unknown") for r in scope3_results or [])
                )
            ][:5]
            if scope3_results
            else [],
            "total_activities": len(scope12_results) + len(scope3_results or []),
        }

        prompt = f"""
You are a comprehensive carbon footprint expert and sustainability consultant. Analyze the following complete emissions data covering all three scopes and provide strategic recommendations.

COMPLETE EMISSIONS DATA:
{json.dumps(emissions_summary, indent=2)}

Please provide a comprehensive analysis including:

1. **Overall Carbon Footprint Assessment**
   - Total emissions evaluation and context
   - Scope distribution analysis and implications

2. **Priority Areas for Emission Reduction**
   - Identify highest impact opportunities
   - Compare Scope 1&2 vs Scope 3 reduction potential

3. **Scope-Specific Recommendations**
   - Scope 1&2: Direct and indirect emission reduction strategies
   - Scope 3: Value chain and supplier engagement strategies

4. **Implementation Roadmap**
   - Quick wins (0-6 months)
   - Medium-term initiatives (6-18 months)
   - Long-term transformation (18+ months)

5. **Compliance and Reporting Considerations**
   - Regulatory requirements
   - Voluntary standards (SBTi, CDP, etc.)
   - Disclosure recommendations

6. **ROI and Business Case**
   - Cost-benefit analysis priorities
   - Estimated emission reduction potential
   - Business value creation opportunities

Format your response in markdown with clear headings and actionable bullet points.
"""

        agent_response, _ = invoke_bedrock_agent(prompt)
        return f"## 🤖 AI-Powered Comprehensive Recommendations\n\n{agent_response}"

    except Exception as e:
        logger.error(f"Error generating comprehensive suggestions: {e}")
        return f"## ⚠️ Suggestions Generation Error\n\nCould not generate AI suggestions: {str(e)}"


# --- APP LAUNCH ---
if __name__ == "__main__":
    initialize_globals()
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True,
    )
