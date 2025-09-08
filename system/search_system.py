#!/usr/bin/env python3
"""
Command-line Information Retrieval System - Task 4
Usage: python system/search_system.py <queries_json> <documents_jsonl> <run_output_json>
"""
import json, sys, pathlib

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.text_preprocessing import preprocess
from index.builders import create_all_indexes
from query_processing.query_process import process_query
from ranking.rankers import rank_documents

def _load_docs(path):
    """Load documents from JSONL file."""
    docs, raw = [], []
    seen_ids = set()
    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "id" not in obj or "text" not in obj:
                            print(f"Warning: Line {line_num} missing required fields (id, text)")
                            continue
                        
                        doc_id = obj["id"]
                        if doc_id in seen_ids:
                            print(f"Warning: Duplicate doc_id {doc_id} found, keeping first occurrence")
                            continue
                        
                        seen_ids.add(doc_id)
                        raw.append(obj)
                        docs.append(obj["text"])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)
    return raw, docs

def _load_queries(path):
    """Load queries from JSON file."""
    try:
        with open(path, 'r') as f:
            queries = json.load(f)
        
        # Validate query format
        if not isinstance(queries, list):
            raise ValueError("Queries file must contain a JSON array")
        
        for i, query in enumerate(queries):
            if not isinstance(query, dict) or "qid" not in query or "query" not in query:
                raise ValueError(f"Query {i} missing required fields (qid, query)")
        
        return queries
    except Exception as e:
        print(f"Error loading queries: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python system/search_system.py <queries_json> <documents_jsonl> <run_output_json>")
        print("Example: python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json")
        sys.exit(1)
    
    queries_path, doc_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Load queries and documents
    print(f"Loading queries from: {queries_path}")
    queries = _load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    print(f"Loading documents from: {doc_path}")
    raw_docs, doc_texts = _load_docs(doc_path)
    print(f"Loaded {len(raw_docs)} documents")
        
    # Build unified index package
    cache_dir = pathlib.Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    unified_index_path = cache_dir / "unified_package.pkl.gz"
    

    print("  Creating unified index package...")
    # we should be able to overwrite the index if it already exists
    create_all_indexes(tokenized_docs_list, str(unified_index_path), doc_ids_list)

    
    # Step 5: Write output JSON
    print(f"Writing results to: {output_path}")
    
    # Ensure output directory exists
    output_dir = pathlib.Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Successfully wrote {len(results)} query results")


if __name__ == "__main__":
    main()
