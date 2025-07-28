import json
import subprocess
import argparse

def read_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_context(json_data):
    # Flatten all blocks into a readable string
    return "\n".join(block["text"] for block in json_data if "text" in block and isinstance(block["text"], str))

def ask_phi(query, context):
    prompt = f"""You are a helpful assistant. The user will ask a question about this document.
    
Document:
\"\"\"
{context}
\"\"\"

Question: {query}
Answer:"""

    # Run query using Ollama
    result = subprocess.run(
        ["ollama", "run", "phi"],
        input=prompt.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return result.stdout.decode().strip()

def main():
    parser = argparse.ArgumentParser(description="Ask Phi-2 LLM about a document")
    parser.add_argument('--file', '-f', required=True, help="Path to structured JSON file")
    parser.add_argument('--query', '-q', required=True, help="Your natural language question")
    args = parser.parse_args()

    json_data = read_json_data(args.file)
    context = build_context(json_data)
    answer = ask_phi(args.query, context)

    print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    main()
