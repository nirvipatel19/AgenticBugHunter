import pandas as pd
from agents.retriever_agent import run_retriever
from agents.analyzer_agent import run_analyzer_single


def main():

    print("Loading dataset...\n")

    df = pd.read_csv("documents/samples.csv")
    print(f"Total rows: {len(df)}\n")

    results = []

    for _, row in df.iterrows():

        code_id = row["ID"]
        context = str(row["Context"])
        code = str(row["Code"])

        print("=======================================")
        print(f"Processing ID: {code_id}")
        print("=======================================")

        # Retrieval
        retrieved_docs = run_retriever(context, code)

        docs_text = ""
        score = ""

        if retrieved_docs:
            docs_text = retrieved_docs[0]["text"]
            score = retrieved_docs[0]["score"]

        print("\nRetrieved Documentation:")
        print(f"Score: {score}")
        print(docs_text[:500])
        print("\n")

        # Analysis
        analysis = run_analyzer_single(
            code_id,
            context,
            code,
            docs_text
        )

        print("Final Output:")
        print(analysis)
        print("\n")

        results.append(analysis)

    # Save CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv("output.csv", index=False)

    print("\nDone. output.csv generated.")


if __name__ == "__main__":
    main()
