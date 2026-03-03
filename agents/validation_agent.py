import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)


def validate_explanation(code_id, explanation, docs):

    prompt = f"""
Check if the following explanation is supported by the documentation.

Documentation:
{docs}

Explanation:
{explanation}

Return STRICT JSON:

{{
  "valid": true/false,
  "reason": "<brief reason>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:groq",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.strip("```").strip()

        return json.loads(content)

    except:
        return {"valid": False, "reason": "Validation failed"}
