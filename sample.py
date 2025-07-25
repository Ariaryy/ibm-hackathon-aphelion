import boto3
import json
import uuid

AGENT_ID = "24AZIE3K8M"
AGENT_ALIAS_ID = "U9BKUOINEZ"
REGION = "ap-south-1" 

bedrock = boto3.client("bedrock-agent", region_name=REGION)

response = bedrock.invoke_agent(
    agentId=AGENT_ID,
    agentAliasId=AGENT_ALIAS_ID,
    sessionId=str(uuid.uuid4()),  # Required
    inputText=json.dumps({
        "description": "Uber ride from office to airport",
        "amount_spent_inr": 450,
        "vendor": "Uber",
        "date": "2025-07-20"
    })
)

print(response["completion"])
