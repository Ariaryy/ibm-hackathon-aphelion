import gradio as gr
import pandas as pd
import boto3
import PyPDF2
import json
import os
import logging
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# --- INITIAL SETUP ---
# Load environment variables from .env.local at the very beginning
load_dotenv(dotenv_path=".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & GLOBAL VARIABLES ---
# Load from environment with fallbacks for safety
REGION = os.getenv("REGION", "ap-south-1")
FLOW_ID = os.getenv("FLOW_ID")
FLOW_ALIAS_ID = os.getenv("FLOW_ALIAS_ID")
AGENT_ID = os.getenv("AGENT_ID")  # New: Bedrock Agent ID
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID")  # New: Bedrock Agent Alias ID

# Validate that necessary environment variables are set
if not FLOW_ID or not FLOW_ALIAS_ID:
    logger.error(
        "FATAL: FLOW_ID and FLOW_ALIAS_ID must be set in your .env.local file."
    )

if not AGENT_ID or not AGENT_ALIAS_ID:
    logger.error(
        "FATAL: AGENT_ID and AGENT_ALIAS_ID must be set in your .env.local file."
    )

# --- GLOBAL CLIENTS AND DATA ---
# These are initialized once to be reused across multiple runs
BEDROCK_RUNTIME_CLIENT = None
BEDROCK_AGENT_CLIENT = None  # New: For Bedrock Agent
GHG_FACTORS_DF = None
CURRENT_RESULTS = []  # Store current analysis results for agent context
CURRENT_SESSION_ID = None  # Store agent session for context


def initialize_globals():
    """Initialize global resources like the Boto3 clients and GHG data."""
    global BEDROCK_RUNTIME_CLIENT, BEDROCK_AGENT_CLIENT, GHG_FACTORS_DF

    # Setup Bedrock Flow client
    try:
        if not FLOW_ID or not FLOW_ALIAS_ID:
            raise ValueError("Flow ID or Alias ID is not configured.")
        BEDROCK_RUNTIME_CLIENT = boto3.client(
            "bedrock-agent-runtime", region_name=REGION
        )
        logger.info(
            f"✅ Bedrock flow runtime client initialized successfully for region: {REGION}"
        )
    except Exception as e:
        logger.error(f"⚠️ Failed to configure Bedrock flow client: {e}")

    # Setup Bedrock Agent client (same client, different methods)
    try:
        if not AGENT_ID or not AGENT_ALIAS_ID:
            raise ValueError("Agent ID or Alias ID is not configured.")
        BEDROCK_AGENT_CLIENT = boto3.client("bedrock-agent-runtime", region_name=REGION)
        logger.info(
            f"✅ Bedrock agent runtime client initialized successfully for region: {REGION}"
        )
    except Exception as e:
        logger.error(f"⚠️ Failed to configure Bedrock agent client: {e}")

    # Load GHG conversion factors
    try:
        GHG_FACTORS_DF = pd.read_csv("ghg-conversion-factors-2025-flat-format.csv")
        logger.info(f"✅ Loaded {len(GHG_FACTORS_DF)} GHG conversion factors.")
    except FileNotFoundError:
        logger.error("FATAL: 'ghg-conversion-factors-2025-flat-format.csv' not found.")
    except Exception as e:
        logger.error(f"⚠️ Failed to load GHG conversion factors: {e}")


# --- EXISTING CORE LOGIC (UNCHANGED) ---


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


class GHGFactorMatcher:
    """Simplified matcher for GHG factors using fuzzy logic."""

    def __init__(self, ghg_df: pd.DataFrame):
        if ghg_df is None or ghg_df.empty:
            raise ValueError("GHG DataFrame is not initialized.")
        self.ghg_df = ghg_df
        self.unit_mapping = self._create_unit_mapping()

    def _create_unit_mapping(self) -> Dict[str, str]:
        """Create unit standardization mapping."""
        return {
            "litre": "litres",
            "liter": "litres",
            "l": "litres",
            "litres": "litres",
            "kg": "kg",
            "kilogram": "kg",
            "kgs": "kg",
            "tonne": "tonnes",
            "ton": "tonnes",
            "t": "tonnes",
            "tonnes": "tonnes",
            "kwh": "kWh",
            "mwh": "MWh",
            "kWh": "kWh",
            "km": "km",
            "kilometer": "km",
            "kilometres": "km",
            "cubic meter": "cubic metres",
            "cubic metres": "cubic metres",
            "m3": "cubic metres",
            "cubic meters": "cubic metres",
        }

    def match_activity_to_ghg_factor(self, rag_output: List[Dict]) -> List[Dict]:
        """Match RAG output to GHG database entries with simplified logic."""
        matched_results = []

        for item in rag_output:
            logger.info(f"🔍 Trying to match: {item}")
            matched_item = self._match_single_item(item)
            if matched_item:
                matched_results.append(matched_item)
                logger.info(
                    f"✅ Successfully matched with emissions: {matched_item.get('calculated_emissions', 0)} kg CO₂e"
                )
            else:
                logger.warning(f"❌ Could not match item: {item}")
                # Add unmatched item with zero emissions for transparency
                unmatched_item = item.copy()
                unmatched_item.update(
                    {
                        "calculated_emissions": 0,
                        "match_confidence": 0,
                        "ghg_conversion_factor": 0,
                        "matched_column_text": "No match found",
                        "error": "Could not find matching GHG factor",
                    }
                )
                matched_results.append(unmatched_item)

        return matched_results

    def _match_single_item(self, item: Dict) -> Optional[Dict]:
        """Match a single item with simplified fuzzy matching."""
        activity = str(item.get("activity", "")).lower()
        fuel = str(item.get("fuel", "")).lower()
        unit = str(item.get("unit", "")).lower()
        scope = str(item.get("scope", "")).replace("Scope ", "").strip()
        quantity = float(item.get("quantity", 0))

        # Standardize unit
        standard_unit = self.unit_mapping.get(unit, unit)

        logger.info(
            f"   Activity: {activity}, Fuel: {fuel}, Unit: {standard_unit}, Scope: {scope}"
        )

        # Filter by scope first
        scope_matches = self.ghg_df[
            self.ghg_df["Scope"].str.contains(f"Scope {scope}", na=False, case=False)
        ]

        logger.info(f"   Found {len(scope_matches)} scope matches")

        if scope_matches.empty:
            return None

        best_match = None
        best_score = 0

        # Search through scope matches
        for _, row in scope_matches.iterrows():
            score = 0

            # Get searchable text from the row
            column_text = str(row.get("Column Text", "")).lower()
            level_1 = str(row.get("Level 1", "")).lower()
            level_2 = str(row.get("Level 2", "")).lower()
            level_3 = str(row.get("Level 3", "")).lower()
            row_unit = str(row.get("UOM", "")).lower()

            # Combine all searchable text
            searchable_text = f"{column_text} {level_1} {level_2} {level_3}".lower()

            # Score based on fuel type matching
            fuel_keywords = {
                "diesel": ["diesel", "gasoil", "gas oil"],
                "petrol": ["petrol", "gasoline", "motor gasoline"],
                "electricity": ["electricity", "electric", "grid"],
                "natural gas": ["natural gas", "gas", "lng", "cng"],
                "coal": ["coal", "anthracite", "bituminous"],
            }

            # Check fuel type matching
            if fuel in fuel_keywords:
                for keyword in fuel_keywords[fuel]:
                    if keyword in searchable_text:
                        score += 40
                        break
            else:
                # Direct fuel matching
                if fuel and fuel in searchable_text:
                    score += 30

            # Check activity matching
            activity_keywords = activity.split()
            for keyword in activity_keywords:
                if len(keyword) > 2 and keyword in searchable_text:
                    score += 15

            # Unit matching (very important)
            if standard_unit and row_unit:
                if standard_unit == row_unit:
                    score += 25
                elif standard_unit in row_unit or row_unit in standard_unit:
                    score += 15

            # Special case matching for common patterns
            if "electricity" in fuel and "electricity" in level_1:
                score += 20
            if "fuel" in activity and "fuel" in level_1:
                score += 15
            if "combustion" in activity and (
                "combustion" in searchable_text or "fuel" in level_1
            ):
                score += 15

            # Update best match if this is better
            if score > best_score and score > 30:  # Lower threshold for better matching
                best_score = score
                best_match = row

        if best_match is not None:
            try:
                conversion_factor = float(best_match["GHG Conversion Factor 2025"])
                emissions = quantity * conversion_factor

                enhanced_item = item.copy()
                enhanced_item.update(
                    {
                        "database_id": best_match["ID"],
                        "matched_scope": best_match["Scope"],
                        "level_1": best_match["Level 1"],
                        "level_2": best_match["Level 2"],
                        "level_3": best_match["Level 3"],
                        "level_4": best_match.get("Level 4", ""),
                        "matched_column_text": best_match["Column Text"],
                        "matched_unit": best_match["UOM"],
                        "ghg_conversion_factor": conversion_factor,
                        "ghg_unit": best_match["GHG/Unit"],
                        "calculated_emissions": round(emissions, 4),
                        "match_confidence": round(best_score / 100, 2),
                    }
                )

                logger.info(
                    f"   ✅ Match found: {best_match['Column Text']} (score: {best_score})"
                )
                return enhanced_item

            except (ValueError, KeyError) as e:
                logger.error(f"   ⚠️ Error processing match: {e}")
                return None

        logger.info(f"   ❌ No suitable match found (best score: {best_score})")
        return None


def invoke_bedrock_flow(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Send text to Bedrock Flow and get structured JSON response.
    [EXISTING IMPLEMENTATION - UNCHANGED]
    """
    if not BEDROCK_RUNTIME_CLIENT:
        raise ConnectionError("Bedrock client not available. Cannot invoke flow.")

    if not text or not text.strip():
        raise ValueError("Input text is empty or None")

    logger.info(f"🚀 Invoking Bedrock Flow with input length: {len(text)} characters")

    try:
        response = BEDROCK_RUNTIME_CLIENT.invoke_flow(
            flowIdentifier=FLOW_ID,
            flowAliasIdentifier=FLOW_ALIAS_ID,
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
                parsed_result = json.loads(raw_json_str)
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


# --- NEW: BEDROCK AGENT INTEGRATION ---


def invoke_bedrock_agent(prompt: str, session_id: str = None) -> Tuple[str, str]:
    """
    Invoke Bedrock Agent for suggestions and chatbot responses.

    Args:
        prompt (str): The prompt to send to the agent
        session_id (str): Session ID for maintaining context (optional)

    Returns:
        Tuple[str, str]: (agent_response, session_id)
    """
    if not BEDROCK_AGENT_CLIENT:
        raise ConnectionError("Bedrock Agent client not available.")

    try:
        # Generate session ID if not provided
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        logger.info(f"🤖 Invoking Bedrock Agent with session: {session_id[:8]}...")

        logger.info(f"Prompt for analysis: {prompt}")

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


def generate_suggestions_with_agent(results: List[Dict[str, Any]]) -> str:
    """Generate AI-powered suggestions using Bedrock Agent."""
    if not results:
        return "## 💡 AI Suggestions\nNo analysis results available to generate suggestions."

    try:
        # Prepare emissions data summary for the agent
        total_emissions = sum(res.get("calculated_emissions", 0) for res in results)
        scope_totals = {"1": 0, "2": 0, "3": 0}

        for res in results:
            scope = str(res.get("scope", "")).replace("Scope ", "").strip()
            if scope in scope_totals:
                scope_totals[scope] += res.get("calculated_emissions", 0)

        # Get top 5 activities
        top_activities = sorted(
            results, key=lambda x: x.get("calculated_emissions", 0), reverse=True
        )[:5]

        # Create structured prompt for the agent
        emissions_summary = {
            "total_emissions_kg_co2e": total_emissions,
            "total_emissions_tonnes_co2e": total_emissions / 1000,
            "scope_breakdown": {
                "scope_1": scope_totals["1"],
                "scope_2": scope_totals["2"],
                "scope_3": scope_totals["3"],
            },
            "top_activities": [
                {
                    "activity": act.get("activity", "Unknown"),
                    "emissions_kg_co2e": act.get("calculated_emissions", 0),
                    "scope": act.get("scope", "Unknown"),
                    "percentage_of_total": (
                        act.get("calculated_emissions", 0) / total_emissions * 100
                    )
                    if total_emissions > 0
                    else 0,
                }
                for act in top_activities
            ],
            "total_activities": len(results),
        }

        prompt = f"""
You are a carbon footprint expert and sustainability consultant. Analyze the following emissions data and provide actionable suggestions for reducing carbon footprint and ensuring compliance with environmental regulations.

EMISSIONS DATA:
{json.dumps(emissions_summary, indent=2)}

Please provide:
1. Overall assessment of the carbon footprint
2. Priority areas for emission reduction (focus on highest impact)
3. Specific actionable recommendations for each scope
4. Compliance considerations and regulatory guidance
5. Quick wins vs long-term strategies
6. Estimated potential emission reductions

Format your response in markdown with clear headings and bullet points. Be specific and actionable.
"""

        agent_response, _ = invoke_bedrock_agent(prompt)
        return f"## 🤖 AI-Powered Suggestions\n\n{agent_response}"

    except Exception as e:
        logger.error(f"Error generating agent suggestions: {e}")
        return f"## ⚠️ Suggestions Generation Error\n\nCould not generate AI suggestions: {str(e)}"


def chatbot_with_agent(message: str, history: List[List[str]]) -> List[List[str]]:
    """Handle chatbot interactions using Bedrock Agent with emissions context."""
    global CURRENT_RESULTS, CURRENT_SESSION_ID

    if not message.strip():
        return history

    try:
        # Prepare context about current emissions data
        context_prompt = ""
        if CURRENT_RESULTS:
            total_emissions = sum(
                res.get("calculated_emissions", 0) for res in CURRENT_RESULTS
            )
            scope_totals = {"1": 0, "2": 0, "3": 0}

            for res in CURRENT_RESULTS:
                scope = str(res.get("scope", "")).replace("Scope ", "").strip()
                if scope in scope_totals:
                    scope_totals[scope] += res.get("calculated_emissions", 0)

            context_prompt = f"""
CONTEXT: The user has analyzed their carbon footprint with the following results:
- Total emissions: {total_emissions:,.2f} kg CO₂e ({total_emissions / 1000:.2f} tonnes)
- Scope 1: {scope_totals["1"]:,.2f} kg CO₂e
- Scope 2: {scope_totals["2"]:,.2f} kg CO₂e  
- Scope 3: {scope_totals["3"]:,.2f} kg CO₂e
- Total activities: {len(CURRENT_RESULTS)}

User's question: {message}

Please provide a helpful response based on their specific emissions data. Be conversational and refer to their actual numbers when relevant.
"""
        else:
            context_prompt = f"""
The user hasn't uploaded any emissions data yet. 

User's question: {message}

Please provide general guidance about carbon footprint analysis and encourage them to upload a report for personalized insights.
"""

        # Get response from agent
        agent_response, session_id = invoke_bedrock_agent(
            context_prompt, CURRENT_SESSION_ID
        )

        # Update session ID for context continuity
        CURRENT_SESSION_ID = session_id

        # Add to history
        history.append([message, agent_response])

        return history

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        error_message = "I'm having trouble processing your request right now. Please try again or check the system logs."
        history.append([message, error_message])
        return history


# --- VISUALIZATION FUNCTIONS ---
def create_emissions_charts(
    results: List[Dict[str, Any]],
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Create comprehensive visualization charts from emissions data."""
    if not results:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available", x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, empty_fig, empty_fig, empty_fig

    logger.info(f"📊 Creating charts for {len(results)} results")

    # Filter out results with zero emissions for main charts, but keep them for debugging
    valid_results = [r for r in results if r.get("calculated_emissions", 0) > 0]
    logger.info(f"📊 {len(valid_results)} results have calculated emissions > 0")

    if not valid_results:
        # If no valid results, show what we have anyway
        logger.warning("⚠️ No results with emissions > 0, showing all data")
        valid_results = results

    # 1. Scope Overview Chart
    scope_totals = {"Scope 1": 0, "Scope 2": 0, "Scope 3": 0}

    for res in valid_results:
        scope_key = res.get("scope", "").strip()
        if not scope_key.startswith("Scope"):
            scope_key = f"Scope {scope_key}"

        emissions = float(res.get("calculated_emissions", 0))
        if scope_key in scope_totals:
            scope_totals[scope_key] += emissions

    # Remove zero values for cleaner display
    filtered_scope_data = {k: v for k, v in scope_totals.items() if v > 0}

    if not filtered_scope_data:
        # Fallback: show all activities regardless of scope
        filtered_scope_data = {
            "Total Emissions": sum(
                r.get("calculated_emissions", 0) for r in valid_results
            )
        }

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    scope_fig = go.Figure(
        data=[
            go.Bar(
                x=list(filtered_scope_data.keys()),
                y=list(filtered_scope_data.values()),
                marker_color=colors[: len(filtered_scope_data)],
                text=[f"{v:,.1f} kg CO₂e" for v in filtered_scope_data.values()],
                textposition="auto",
            )
        ]
    )

    total_emissions = sum(filtered_scope_data.values())
    scope_fig.update_layout(
        title=f"Emissions by Scope - Total: {total_emissions:,.2f} kg CO₂e",
        xaxis_title="Emission Scopes",
        yaxis_title="Emissions (kg CO₂e)",
        height=400,
    )

    # 2. Top Activities Chart - Show all activities
    # Sort by emissions (including zero emissions for transparency)
    sorted_activities = sorted(
        results, key=lambda x: x.get("calculated_emissions", 0), reverse=True
    )
    top_activities = sorted_activities[:10]  # Top 10

    activities_fig = go.Figure(
        data=[
            go.Bar(
                y=[
                    f"{act.get('activity', 'Unknown')[:30]}..."
                    if len(str(act.get("activity", ""))) > 30
                    else str(act.get("activity", "Unknown"))
                    for act in top_activities
                ],
                x=[act.get("calculated_emissions", 0) for act in top_activities],
                orientation="h",
                marker_color=[
                    "#FF6B6B" if x.get("calculated_emissions", 0) > 0 else "#FFB6B6"
                    for x in top_activities
                ],
                text=[
                    f"{act.get('calculated_emissions', 0):,.2f}"
                    for act in top_activities
                ],
                textposition="auto",
            )
        ]
    )

    activities_fig.update_layout(
        title="Top 10 Activities by Emissions",
        xaxis_title="Emissions (kg CO₂e)",
        yaxis_title="Activities",
        height=500,
        margin=dict(l=200),
    )

    # 3. Scope Distribution Pie Chart (only if we have scope data)
    if len(filtered_scope_data) > 1:
        pie_fig = go.Figure(
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

        pie_fig.update_layout(title="Emissions Distribution by Scope", height=400)
    else:
        # If only one scope or no scope data, show activity distribution
        activity_emissions = [
            (r.get("activity", "Unknown"), r.get("calculated_emissions", 0))
            for r in valid_results[:8]
        ]  # Top 8 activities

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        item[0][:20] + "..." if len(item[0]) > 20 else item[0]
                        for item in activity_emissions
                    ],
                    values=[item[1] for item in activity_emissions],
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>"
                    + "Emissions: %{value:.2f} kg CO₂e<br>"
                    + "<extra></extra>",
                )
            ]
        )

        pie_fig.update_layout(title="Top Activities Distribution", height=400)

    # 4. Fuel Type Analysis
    fuel_emissions = {}
    for res in valid_results:
        fuel = res.get("fuel", "Unknown")
        emissions = res.get("calculated_emissions", 0)
        fuel_emissions[fuel] = fuel_emissions.get(fuel, 0) + emissions

    # Sort by emissions
    sorted_fuels = sorted(fuel_emissions.items(), key=lambda x: x[1], reverse=True)[:8]

    if sorted_fuels:
        category_fig = go.Figure(
            data=[
                go.Bar(
                    x=[fuel[0] for fuel in sorted_fuels],
                    y=[fuel[1] for fuel in sorted_fuels],
                    marker_color="#45B7D1",
                    text=[f"{fuel[1]:,.2f}" for fuel in sorted_fuels],
                    textposition="auto",
                )
            ]
        )

        category_fig.update_layout(
            title="Emissions by Fuel Type",
            xaxis_title="Fuel Types",
            yaxis_title="Emissions (kg CO₂e)",
            xaxis_tickangle=-45,
            height=400,
        )
    else:
        # Fallback: show match status
        matched_count = len(
            [r for r in results if r.get("calculated_emissions", 0) > 0]
        )
        unmatched_count = len(results) - matched_count

        category_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Successfully Matched", "Not Matched"],
                    y=[matched_count, unmatched_count],
                    marker_color=["#4ECDC4", "#FF6B6B"],
                    text=[str(matched_count), str(unmatched_count)],
                    textposition="auto",
                )
            ]
        )

        category_fig.update_layout(
            title="Matching Status",
            xaxis_title="Status",
            yaxis_title="Number of Activities",
            height=400,
        )

    logger.info("✅ Charts created successfully")
    return scope_fig, activities_fig, pie_fig, category_fig


def debug_matching_process(results: List[Dict[str, Any]]) -> str:
    """Create a debug report showing matching results."""
    if not results:
        return "## 🔍 Debug Report\nNo results to analyze."

    total_items = len(results)
    matched_items = len([r for r in results if r.get("calculated_emissions", 0) > 0])
    unmatched_items = total_items - matched_items

    total_emissions = sum(r.get("calculated_emissions", 0) for r in results)

    debug_report = f"""
## 🔍 Debug Report

### Summary
- **Total Activities**: {total_items}
- **Successfully Matched**: {matched_items}
- **Not Matched**: {unmatched_items}
- **Total Emissions**: {total_emissions:,.2f} kg CO₂e

### Detailed Results
"""

    for i, result in enumerate(results, 1):
        activity = result.get("activity", "Unknown")
        fuel = result.get("fuel", "Unknown")
        quantity = result.get("quantity", 0)
        unit = result.get("unit", "Unknown")
        emissions = result.get("calculated_emissions", 0)
        confidence = result.get("match_confidence", 0)
        matched_text = result.get("matched_column_text", "No match")

        status = "✅ Matched" if emissions > 0 else "❌ Not Matched"

        debug_report += f"""
**{i}. {activity}**
- Fuel: {fuel}
- Quantity: {quantity} {unit}
- Emissions: {emissions:,.2f} kg CO₂e
- Confidence: {confidence:.2f}
- Matched to: {matched_text}
- Status: {status}
"""

    return debug_report


# --- MAIN ANALYSIS FUNCTION (EXTENDED) ---


def run_analysis(pdf_file):
    """Enhanced analysis function with better error handling and debugging."""
    global CURRENT_RESULTS, CURRENT_SESSION_ID

    if not all(
        [BEDROCK_RUNTIME_CLIENT, BEDROCK_AGENT_CLIENT, GHG_FACTORS_DF is not None]
    ):
        error_msg = "## ⚠️ Error: Application not initialized. Check console logs."
        logger.error("Application components not initialized properly")
        return (error_msg, None, None, None, None, None, None, error_msg)

    if pdf_file is None:
        return (
            "## Please upload a PDF file first.",
            None,
            None,
            None,
            None,
            None,
            None,
            "No PDF file provided",
        )

    try:
        logger.info(f"🚀 Starting analysis for PDF: {pdf_file.name}")

        # Step 1: Extract text from PDF
        input_text = extract_text_from_pdf(pdf_file.name)
        logger.info(f"📄 Extracted {len(input_text)} characters from PDF")

        # Step 2: Process with Bedrock Flow
        rag_results, raw_json_str = invoke_bedrock_flow(input_text)
        logger.info(f"🔍 Flow returned {len(rag_results)} activities")

        # Log what we got from the flow
        for i, result in enumerate(rag_results):
            logger.info(
                f"   Activity {i + 1}: {result.get('activity', 'Unknown')} - "
                f"{result.get('quantity', 0)} {result.get('unit', 'Unknown')} - "
                f"Scope {result.get('scope', 'Unknown')}"
            )

        # Step 3: Match to GHG factors
        matcher = GHGFactorMatcher(GHG_FACTORS_DF)
        final_results = matcher.match_activity_to_ghg_factor(rag_results)

        # Log matching results
        matched_count = len(
            [r for r in final_results if r.get("calculated_emissions", 0) > 0]
        )
        total_emissions = sum(r.get("calculated_emissions", 0) for r in final_results)
        logger.info(f"🎯 Matched {matched_count}/{len(final_results)} activities")
        logger.info(f"💨 Total emissions: {total_emissions:,.2f} kg CO₂e")

        # Store results for agent context
        CURRENT_RESULTS = final_results
        CURRENT_SESSION_ID = None  # Reset session for new analysis

        # Step 4: Create visualizations (always create, even with zero data)
        logger.info("📊 Creating visualizations...")
        try:
            scope_chart, activities_chart, pie_chart, category_chart = (
                create_emissions_charts(final_results)
            )
            logger.info("✅ Charts created successfully")
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            # Create empty charts as fallback
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Chart error: {str(e)}", x=0.5, y=0.5, showarrow=False
            )
            scope_chart = activities_chart = pie_chart = category_chart = empty_fig

        # Step 5: Generate AI suggestions
        logger.info("🤖 Generating AI-powered suggestions...")
        try:
            if matched_count > 0:
                suggestions = generate_suggestions_with_agent(final_results)
            else:
                suggestions = f"""
## 🤖 AI Analysis Results

### ⚠️ Matching Issues Detected

I found {len(final_results)} activities in your report, but could not calculate emissions for any of them due to matching issues with the GHG database.

**Activities Found:**
{chr(10).join([f"- {r.get('activity', 'Unknown')}: {r.get('quantity', 0)} {r.get('unit', 'Unknown')} (Scope {r.get('scope', 'Unknown')})" for r in final_results[:5]])}

### 🔧 Possible Solutions:
1. **Check Unit Formats**: Ensure units match expected formats (litres, kg, kWh, etc.)
2. **Verify Scope Classification**: Activities should be clearly marked as Scope 1, 2, or 3
3. **Review Activity Descriptions**: Use standard terminology (e.g., "diesel combustion", "electricity consumption")
4. **Database Coverage**: Some specific fuels or activities might not be in the UK GHG database

### 📋 Debug Information:
{debug_matching_process(final_results)}
"""
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            suggestions = (
                f"## ⚠️ Suggestions Error\n\nCould not generate AI suggestions: {str(e)}"
            )

        # Step 6: Prepare results DataFrame
        try:
            results_df = pd.DataFrame(final_results)
            if not results_df.empty:
                # Select key columns for display
                display_columns = [
                    "activity",
                    "fuel",
                    "quantity",
                    "unit",
                    "scope",
                    "calculated_emissions",
                    "match_confidence",
                    "matched_column_text",
                ]
                # Only include columns that exist
                available_columns = [
                    col for col in display_columns if col in results_df.columns
                ]
                results_df = results_df[available_columns]
        except Exception as e:
            logger.error(f"DataFrame creation error: {e}")
            results_df = pd.DataFrame(
                [{"Error": f"Could not create results table: {str(e)}"}]
            )

        # Step 7: Save JSON file
        json_path = "carbon_footprint_results.json"
        try:
            with open(json_path, "w") as f:
                json.dump(final_results, f, indent=2)
            download_file = gr.File(value=json_path, visible=True)
        except Exception as e:
            logger.error(f"JSON save error: {e}")
            download_file = None

        logger.info("✅ Analysis complete!")

        return (
            suggestions,  # AI-generated suggestions or debug info
            scope_chart,  # Scope overview chart
            activities_chart,  # Top activities chart
            pie_chart,  # Pie chart
            category_chart,  # Category chart
            results_df,  # Detailed results table
            download_file,  # Download file
            None,  # No error
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        error_message = f"## ❌ Analysis Error\n\n**Error:** `{str(e)}`\n\n**Please check:**\n- PDF file is readable\n- Bedrock services are configured\n- GHG database is loaded"

        # Return empty charts on error
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Analysis failed", x=0.5, y=0.5, showarrow=False)

        return (
            error_message,  # Error in suggestions
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,  # Empty charts
            None,  # No results table
            None,  # No download file
            str(e),  # Error message
        )


# --- GRADIO UI (ENHANCED) ---


def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(), title="AI Carbon Footprint Analyzer"
    ) as demo:
        gr.Markdown("# 🌍 AI-Powered Carbon Footprint Analyzer")
        gr.Markdown(
            "Upload a PDF report to extract activities, calculate emissions, and get AI-powered suggestions for reduction strategies."
        )

        # --- PDF UPLOAD SECTION ---
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="📄 Upload Carbon Footprint Report", file_types=[".pdf"]
                )
                process_button = gr.Button(
                    "🚀 Analyze Report", variant="primary", size="lg"
                )
                error_output = gr.Markdown(value="", visible=True)

        # --- MAIN CONTENT TABS ---
        with gr.Tabs():
            # Tab 1: AI Suggestions
            with gr.TabItem("🤖 AI Suggestions"):
                suggestions_output = gr.Markdown(
                    value="## 💡 AI-Powered Suggestions\n\nUpload and process a report to get personalized AI recommendations for reducing your carbon footprint."
                )

            # Tab 2: Interactive Charts
            with gr.TabItem("📊 Visualizations"):
                with gr.Row():
                    scope_plot = gr.Plot(label="Emissions by Scope")
                    pie_plot = gr.Plot(label="Scope Distribution")

                with gr.Row():
                    activities_plot = gr.Plot(label="Top Activities")
                    category_plot = gr.Plot(label="Emissions by Category")

            # Tab 3: AI Chatbot
            with gr.TabItem("💬 AI Assistant"):
                gr.Markdown("## 🤖 Chat with AI Carbon Expert")
                gr.Markdown(
                    "Ask questions about your emissions data, get personalized advice, or learn about carbon footprint management."
                )

                chatbot = gr.Chatbot(
                    value=[],
                    label="AI Carbon Footprint Expert",
                    show_label=True,
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/147/147144.png",
                        "https://cdn-icons-png.flaticon.com/512/4712/4712139.png",
                    ),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask about your carbon footprint",
                        placeholder="e.g., What are my highest emission sources? How can I reduce Scope 2 emissions?",
                        scale=4,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_button = gr.Button("🗑️ Clear Chat", variant="secondary")

            # Tab 4: Detailed Data
            with gr.TabItem("📋 Detailed Results"):
                results_output = gr.DataFrame(
                    headers=[
                        "activity",
                        "quantity",
                        "unit",
                        "scope",
                        "calculated_emissions",
                        "matched_column_text",
                        "ghg_conversion_factor",
                        "match_confidence",
                    ],
                    label="Extracted and Matched Activities",
                )

                with gr.Row():
                    download_output = gr.File(
                        label="📥 Download Full Results (JSON)", visible=False
                    )

        # --- EVENT HANDLERS ---

        # Main analysis process
        process_button.click(
            fn=run_analysis,
            inputs=[pdf_input],
            outputs=[
                suggestions_output,  # AI suggestions
                scope_plot,  # Scope chart
                activities_plot,  # Activities chart
                pie_plot,  # Pie chart
                category_plot,  # Category chart
                results_output,  # Results table
                download_output,  # Download file
                error_output,  # Error messages
            ],
            show_progress=True,
        )

        # Chatbot interactions
        def send_message(message, history):
            if message.strip():
                updated_history = chatbot_with_agent(message, history)
                return "", updated_history
            return message, history

        def clear_chat():
            global CURRENT_SESSION_ID
            CURRENT_SESSION_ID = None  # Reset session
            return []

        send_button.click(
            fn=send_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )

        msg_input.submit(
            fn=send_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )

        clear_chat_button.click(fn=clear_chat, outputs=[chatbot])

        # Add some helpful examples
        gr.Markdown("""
        ## 🚀 Getting Started
        
        1. **Upload PDF**: Upload your carbon footprint or sustainability report
        2. **Analyze**: Click "Analyze Report" to extract emissions data
        3. **Review Suggestions**: Check the AI-powered recommendations in the first tab
        4. **Explore Charts**: Visualize your emissions breakdown in the second tab
        5. **Ask Questions**: Use the AI chatbot for personalized insights and advice
        6. **Download Data**: Export detailed results for further analysis
        
        ## 📝 Example Questions for the AI Assistant
        - "What are my top 3 emission sources?"
        - "How can I reduce my Scope 2 emissions?"
        - "What percentage of my emissions come from transportation?"
        - "Suggest some quick wins for emission reduction"
        - "How do my emissions compare to industry benchmarks?"
        - "What compliance requirements should I be aware of?"
        """)

    return demo


# --- APP LAUNCH ---
if __name__ == "__main__":
    initialize_globals()
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
    )
