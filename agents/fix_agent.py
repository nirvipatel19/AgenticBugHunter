"""
Fix Suggestion Agent
--------------------
Uses Gemini to suggest corrected code.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-lite",
    temperature=0
)


def run_fix_agent(code: str, bug_line: int, explanation: str):
    """
    Suggests corrected version of buggy code.

    Parameters:
        code (str): Original buggy snippet.
        bug_line (int): Line number of bug.
        explanation (str): Explanation from analyzer.

    Returns:
        str: Corrected code suggestion.
    """

    prompt = ChatPromptTemplate.from_template("""
You are a C++ expert.

Given:
- Buggy code
- The bug line number
- The explanation of the bug

Provide the corrected version of the buggy line or snippet.

Rules:
- Do NOT explain.
- Return only corrected code.
- Preserve original formatting if possible.

Buggy Code:
{code}

Bug Line:
{bug_line}

Bug Explanation:
{explanation}
""")

    chain = prompt | llm

    response = chain.invoke({
        "code": code,
        "bug_line": bug_line,
        "explanation": explanation
    })

    return response.content.strip()
