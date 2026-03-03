import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def run_analyzer_single(code_id, context, code, docs):

    # CHANGED: Refined prompt to prevent C++ syntax hallucinations
    prompt = f"""
You are a SmartRDI Hardware API verification expert. Your goal is to detect LOGICAL errors in test program code based strictly on the provided documentation.

**CRITICAL RULES TO AVOID HALLUCINATIONS:**
1. **IGNORE C++ SYNTAX:** This environment uses custom macros. 
   - `30 V`, `100 uA`, `100 us` are VALID custom literals. Do NOT flag them as syntax errors.
   - `wait(100 us)` is VALID. Do NOT flag as a type mismatch.
2. **CHECK VALUES:** Check if numeric values exceed limits defined in the documentation (e.g., if Doc says "Max 30V" and code has "31 V", that is the bug).
3. **CHECK LIFECYCLE:** Verify that commands are inside/outside `RDI_BEGIN` and `RDI_END` as per documentation.
4. **CHECK MODES:** Look for incorrect enum usage (e.g., `TA::VECD` vs `TA::VTT`).
5. **CHECK TYPOS:** Look for slight misspellings in function names (e.g., `readHumanSeniority` vs `readHumSensor`).

Analyze the following snippet.

ID: {code_id}

Context:
{context}

Code:
{code}

Retrieved Documentation:
{docs}

Instructions:
- Identify the exact bug line number.
- Provide a clear explanation based on the LOGIC or API MISUSE.
- Respond with STRICT JSON ONLY.
- No markdown.
- No extra commentary.

Format:
{{
  "ID": {code_id},
  "bug_line": <integer>,
  "explanation": "<text>"
}}
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b:groq", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        content = completion.choices.message.content.strip()

        # Remove markdown fences if present
        if content.startswith("```"):
            content = content.strip("```").strip()

        parsed = json.loads(content)
        return parsed

    except Exception as e:
        print(f"\n❌ HF Router Error for ID {code_id}:")
        print(str(e))

        return {
            "ID": code_id,
            "bug_line": 1,
            "explanation": "HF Router API error."
        }