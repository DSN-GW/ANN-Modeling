import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv


class SonnetAgent:
    def __init__(self, region_name="us-east-1"):
        load_dotenv()
        
        session = boto3.Session(
            aws_access_key_id=os.environ.get('aws_access_key_id'),
            aws_secret_access_key=os.environ.get('aws_secret_access_key'),
            aws_session_token=os.environ.get('aws_session_token')
        )
        
        self.bedrock_client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.system_prompt = ""
        self.max_tokens = 1000
        self.temperature = 0
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
    
    def set_parameters(self, max_tokens=None, temperature=None):
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
    
    def ask(self, question):
        messages = [
            {
                "role": "user",
                "content": question
            }
        ]
        
        body_content = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if self.system_prompt:
            body_content["system"] = self.system_prompt
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body_content)
        )
        response_body = response['body'].read().decode()
        response_data = json.loads(response_body)
        return response_data.get('content', [{}])[0].get('text', '')
