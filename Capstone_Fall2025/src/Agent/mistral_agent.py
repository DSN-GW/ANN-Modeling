import boto3
from botocore.exceptions import ClientError
import json
import os
from dotenv import load_dotenv


class MistralAgent:
    def __init__(self, region_name="us-east-1"):
        load_dotenv()

        if os.environ.get("AWS_PROFILE"):
            session = boto3.Session(profile_name=os.environ["AWS_PROFILE"], region_name=region_name)
        else:
            session = boto3.Session(
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                region_name=region_name,
            )

        self.bedrock_client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = "deepseek.r1-v1:0"
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
        # Build OpenAI-style messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user",
                         "content": [{"type": "text", "text": question}]})

        body = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Optional fields Qwen3 supports:
            # "top_p": 0.9,
            # "stop": ["\n\n"]
        }

        resp = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        data = json.loads(resp["body"].read())

        # Robust extraction across providers
        # Bedrock OpenAI-style returns choices[0].message.content[0].text
        try:
            choices = data.get("choices", [])
            if choices:
                content = choices[0]["message"]["content"]
                if isinstance(content, list) and content and "text" in content[0]:
                    return content[0]["text"]
                if isinstance(content, str):
                    return content
        except Exception:
            pass

        # Fallbacks for other shapes
        if "output" in data and "message" in data["output"]:
            return "".join(p.get("text", "") for p in data["output"]["message"]["content"])
        if "results" in data and data["results"]:
            return data["results"][0].get("text") or data["results"][0].get("outputText")
        if "outputText" in data:
            return data["outputText"]
        return str(data)



