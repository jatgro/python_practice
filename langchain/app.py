import openai
import os
from dotenv import load_dotenv, find_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import httpx

auth =  "https://api.uhg.com/oauth2/token"
scope = "https://api.uhg.com/.default"
grant_type="client_credentials"


load_dotenv(find_dotenv())

client_secret=os.getenv("AZURE_OPENAI_API_KEY")

# use an asynchronous client to make a POST request to auth url
async with httpx.AsyncClient() as client: 
    body = {
        "grant_type" = grant_type,
        "scope": scope,
        "client_id":"58c7c302-9456-48bf-a099-1c0c0c9e7421",
        "client_secret":client_secret
    }

    headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }

    resp = await client.post(auth, headers, data=body, timeout=60)
    access_token = resp.json()["access_token"]

    deployment_name = "gpt-5_2025-08-07"

    shared_quota-endpoint = "https://api.uhg.com/api/cloud/api-management/ai-gateway-reasoning/1.0"
    azure_openai_api_version = "2025-01-01-preview"

    oai_client = openai.AzureOpenAI(
        azure_endpoint=shared_quota-endpoint, 
        api_version=azure_openai_api_version, 
        azure_deployment=deployment_name, 
        azure_ad_token=access_token, 
        default_headers={
            "projectId":"29396466-5b23-4334-a7a3-d072a722ca06",
        }
    )

    messages = [{'role':"user", 'content':"hi, what is a prime number"}]
    
    response = oai_client.chat.completions.create(
        model="gpt-5", 
        messages=messages
    )

    print(response.model_dump_json(indent=2))
