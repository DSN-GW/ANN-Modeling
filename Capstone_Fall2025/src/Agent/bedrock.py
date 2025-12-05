# list_accessible_bedrock_models.py
import os, json, boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
region = os.getenv("AWS_REGION", "us-east-1")

# --- Session (profile or static creds) ---
if os.environ.get("AWS_PROFILE"):
    session = boto3.Session(profile_name=os.environ["AWS_PROFILE"], region_name=region)
else:
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=region,
    )

br = session.client("bedrock", region_name=region)               # control plane
rt = session.client("bedrock-runtime", region_name=region)       # runtime (Converse / InvokeModel)

# Get the regional catalog (text, on-demand models)
resp = br.list_foundation_models(byOutputModality="TEXT", byInferenceType="ON_DEMAND")
models = resp.get("modelSummaries", [])

def has_inference_access(model_id: str) -> bool:
    """
    Returns True if we can successfully call the model (i.e., access granted),
    False if Bedrock returns AccessDeniedException. We keep the request tiny.
    """
    try:
        rt.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1, "temperature": 0},
        )
        return True    # invocation worked => you have access
    except ClientError as e:
        code = e.response["Error"].get("Code", "")
        # Lack of model entitlement shows up as AccessDeniedException on invoke.
        if code == "AccessDeniedException":
            return False
        # Some models may complain (ValidationException, etc.) but that still proves access.
        if code in {"ValidationException", "ThrottlingException", "InternalServerException"}:
            return True
        # Anything else: treat as no-access to be conservative
        return False

accessible = []
for m in models:
    mid = m["modelId"]
    if has_inference_access(mid):
        accessible.append(m)

print(f"\nModels you can invoke in {region}:\n")
for m in accessible:
    print(f"modelId: {m['modelId']}  |  provider: {m.get('providerName')}  |  name: {m.get('modelName')}")
