"""
Retriever Agent
---------------
Uses MCP's prebuilt vector index to retrieve
relevant documentation for a given C++ snippet.
"""

from server.mcp_server import retriever


def run_retriever(context: str, code: str):
    """
    Retrieves top 3 relevant documentation chunks from MCP.

    Parameters:
        context (str): Context provided in dataset.
        code (str): Buggy C++ code snippet.

    Returns:
        list: List of dictionaries with 'text' and 'score'.
    """

    # Build structured semantic query
    query = f"""
    C++ Bug Detection Task

    Context:
    {context}

    Buggy Code Snippet:
    {code}

    Retrieve documentation related to:
    - Incorrect RDI usage
    - Function naming mistakes
    - Parameter mismatch
    - Syntax errors
    - API misuse
    """

    try:
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes[:3]:
            results.append({
                "text": node.get_text(),
                "score": node.get_score()
            })

        return results

    except Exception as e:
        print("Retriever error:", e)
        return []
