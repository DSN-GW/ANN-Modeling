import boto3, os, json
from dotenv import load_dotenv

load_dotenv()
region_name="us-east-1"
# Create a session using your env/profile
if os.environ.get("AWS_PROFILE"):
    session = boto3.Session(profile_name=os.environ["AWS_PROFILE"], region_name=region_name)
else:
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=region_name,
    )

br = session.client("bedrock")

resp = br.list_foundation_models()

print("Available Bedrock models in this region:\n")
for m in resp["modelSummaries"]:
    print(f"modelId: {m['modelId']}  |  provider: {m.get('providerName')}  |  name: {m.get('modelName')}")