from dotenv import load_dotenv
import weaviate
import weaviate.classes as wvc

from benchmark import load_benchmark_cases, print_benchmark_report, run_benchmark
from ingestion.text_ingestion import run_text_ingestion
from ingestion.pdf_transform import run_pdf_transform
from core.log_config import get_logger
from retrieval import run_retrieval

from core.utils import get_genai_client, get_cerebras_client, get_weaviate_client

logger = get_logger(__name__)


def main():
    load_dotenv()
    client = get_genai_client()
    cerebras_client = get_cerebras_client()
    try:
        weaviate_client = get_weaviate_client()
        print("Connected to Weaviate.")
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        weaviate_client = None

    COLLECTION_NAME = "GovDocs"

    RUN_INGESTION = False
    RUN_RETRIEVAL = False
    RUN_BENCHMARK = True

    if RUN_INGESTION:
        print("\n" + "=" * 50)
        print("MODULE: INGESTION PIPELINE")
        print("=" * 50)

        if weaviate_client:
            if not weaviate_client.collections.exists("GovDocs"):
                print("Creating Weaviate collection: 'GovDocs'")
                weaviate_client.collections.create(
                    name="GovDocs",
                    properties=[
                        wvc.config.Property(name="translated_text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="child_text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="parent_context", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="document_title", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="doc_number", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="year", data_type=wvc.config.DataType.INT),
                        wvc.config.Property(name="issuing_authority", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="document_category", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="source_filename", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="supersedes", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="references", data_type=wvc.config.DataType.TEXT),
                    ],
                )
            else:
                print("Using existing Weaviate collection: 'GovDocs'")

        # Phase 1: Transform new PDFs in docs/raw/ → docs/parsed/
        run_pdf_transform(raw_dir="docs/raw", output_dir="docs/parsed")

        # Phase 2: Ingest new .en.txt files into Weaviate
        records = run_text_ingestion(client, weaviate_client=weaviate_client, collection_name=COLLECTION_NAME)
        print("Upsert and indexing complete! All vector records are stored in Weaviate.")

    if RUN_BENCHMARK:
        print("\n" + "=" * 50)
        print("MODULE: CORPUS BENCHMARK & EVALUATION HARNESS")
        print("=" * 50)
        cases = load_benchmark_cases(sample_size=100)
        if cases:
            benchmark_report = run_benchmark(client, cerebras_client, weaviate_client, cases, collection_name=COLLECTION_NAME)
            print_benchmark_report(benchmark_report)
        else:
            print("No benchmark cases found. Create a benchmark.json file.")

    if RUN_RETRIEVAL:
        print("\n" + "=" * 50)
        print("MODULE: RETRIEVAL & GENERATION PIPELINE (INTERACTIVE MODE)")
        print("=" * 50)
        print("Type 'exit' or 'quit' to stop testing. Type 'clear' to reset conversation history.")

        chat_history = []

        while True:
            try:
                query = input("\n[Mimir] Ask a question: ")
                if query.strip().lower() in ["exit", "quit"]:
                    print("Exiting Mimir...")
                    break
                if query.strip().lower() == "clear":
                    chat_history.clear()
                    print("[Memory] Conversation history cleared!")
                    continue
                if not query.strip():
                    continue

                retrieval_result = run_retrieval(client, cerebras_client, weaviate_client, query, COLLECTION_NAME, chat_history)

                print(f"\n{'-' * 50}\nAssistant Response:\n")
                full_answer = ""
                if retrieval_result["status"] != "success":
                    full_answer = (
                        retrieval_result.get("response_text")
                        or "I couldn't find any relevant documents to answer your question."
                    )
                    print(full_answer)
                else:
                    response_stream = retrieval_result["answer_stream"]
                    for chunk in response_stream:
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_answer += chunk
                print(f"\n{'-' * 50}")

                chat_history.append({"role": "user", "text": query})
                chat_history.append({"role": "model", "text": full_answer})

            except KeyboardInterrupt:
                print("\nExiting Mimir...")
                break


if __name__ == "__main__":
    main()
