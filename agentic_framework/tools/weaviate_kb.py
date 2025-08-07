import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

# --- Load environment variables ---
load_dotenv()

HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST")
API_KEY = os.getenv("WEAVIATE_API_KEY")


if not HTTP_HOST:
    raise ValueError("Missing WEAVIATE_HTTP_HOST in environment variables")

# ✅ Cloud URL should NOT include ":443" manually
WEAVIATE_URL = f"https://{HTTP_HOST}"

if not API_KEY:
    raise ValueError("WEAVIATE_API_KEY is missing or empty!")

# --- Connect to Weaviate Cloud ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(API_KEY),
)

# --- Access collection ---
collection = client.collections.get("td_1_banking_issue_kb")
# --- Fetch similar issues ---
def fetch_similar_issues(query_vector, top_k=3):
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
    )
    return [
        {
            "issue_id": obj.properties.get("issue_id"),
            "issue_type": obj.properties.get("issue_type"),
            "remediation_plan": obj.properties.get("remediation_plan"),
            "status": obj.properties.get("status")
        }
        for obj in response.objects
    ]
def close_client():
    client.close()
