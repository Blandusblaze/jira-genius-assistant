import json
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

app = FastAPI()

@app.on_event("startup")
def startup_event():
    preload_ollama_model("moondream")

def preload_ollama_model(model_name: str = "moondream"):
    print(f"🚀 Preloading model '{model_name}'...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "Hello", "stream": False},
            timeout=(5, 120)
        )
        if response.status_code == 200:
            print("✅ Model loaded and ready.")
    except Exception as e:
        print(f"❌ Failed to preload model '{model_name}': {e}")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
collection_name = "jira_issues"

tokenizer = AutoTokenizer.from_pretrained("slm-finetuned")
model = AutoModelForCausalLM.from_pretrained("slm-finetuned")
model.eval()

class Comment(BaseModel):
    author: str
    created: str
    comment: str

class JiraIssue(BaseModel):
    id: str
    key: str
    summary: str
    status: str
    statusCategory: str
    issueType: str
    priority: str
    project: str
    created: str
    updated: str
    description: str = ""
    reporter: str
    assignee: Optional[str] = None
    parent: Optional[str] = None
    issue_number: Optional[int] = None
    comments_flattened: Optional[List[Comment]] = []

class QueryInput(BaseModel):
    query: str
    
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "jira_issues"

def process_batch(batch: List[JiraIssue]) -> List[dict]:
    batch_data = []
    for issue in batch:
        comments = issue.comments_flattened or []
        comment_text = "\n".join(
            f"- {c.author} ({c.created}): {c.comment}" for c in comments
        )

        input_text = (
            f"ID: {issue.id}\n"
            f"Key: {issue.key}\n"
            f"Issue Number: {issue.issue_number}\n"
            f"Project: {issue.project}\n"
            f"Issue Type: {issue.issueType}\n"
            f"Priority: {issue.priority}\n"
            f"Status: {issue.status}\n"
            f"Status Category: {issue.statusCategory}\n"
            f"Created: {issue.created}\n"
            f"Updated: {issue.updated}\n"
            f"Reporter: {issue.reporter}\n"
            f"Assignee: {issue.assignee}\n"
            f"Parent: {issue.parent}\n"
            f"Summary: {issue.summary}\n"
            f"Description: {issue.description}\n"
            f"Comments:\n{comment_text}"
        )

        output_text = (
            f"This is a Jira issue with the following details:\n"
            f"- ID: {issue.id}\n"
            f"- Key: {issue.key}\n"
            f"- Issue Number: {issue.issue_number}\n"
            f"- Project: {issue.project}\n"
            f"- Issue Type: {issue.issueType}\n"
            f"- Priority: {issue.priority}\n"
            f"- Status: {issue.status}\n"
            f"- Status Category: {issue.statusCategory}\n"
            f"- Created on: {issue.created}\n"
            f"- Updated on: {issue.updated}\n"
            f"- Reporter: {issue.reporter}\n"
            f"- Assignee: {issue.assignee}\n"
            f"- Parent: {issue.parent}\n"
            f"- Summary: {issue.summary}\n"
            f"- Description: {issue.description}\n"
            f"- Comments:\n{comment_text}"
        )

        batch_data.append({
            "input": input_text,
            "output": output_text,
            "meta": {
                "key": issue.key,
                "status": issue.status,
                "project": issue.project,
                "priority": issue.priority,
                "type": issue.issueType
            }
        })
    return batch_data


@app.post("/generate_training")
async def generate_training(issues: List[JiraIssue], request: Request):
    print(f"🟡 Received {len(issues)} Jira issues")

    raw_body = await request.json()
    expected_total = raw_body[0].get("total", 0)
    print(f"🟡 Received {len(issues)} Jira issues out of expected total: {expected_total}")

    if not issues:
        return {"error": "No issues provided."}

    batch_size = 100
    batches = [issues[i:i + batch_size] for i in range(0, len(issues), batch_size)]
    print(f"🟡 Split into {len(batches)} batches of size {batch_size}")

    training_file = Path("training_data_jira.json")
    training_data_total = 0

    def safe_process(idx, batch):
        try:
            return process_batch(batch)
        except Exception as e:
            print(f"❌ Error in batch {idx+1}: {e}")
            return []

    all_new_items = []
    for idx, batch in enumerate(batches):
        batch_result = safe_process(idx, batch)
        all_new_items.extend(batch_result)
        training_data_total += len(batch_result)

    if training_file.exists():
        try:
            with training_file.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(all_new_items)
    with training_file.open("w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"💾 All training data saved to {training_file.absolute()}")

    return {
        "message": f"✅ Successfully generated training data for {training_data_total} issues in {len(batches)} batches",
        "training_file": str(training_file.resolve()),
        "total": expected_total,
        "processed": len(existing_data),
        "done": len(existing_data) >= expected_total
    }

    

@app.post("/prepare_training")
async def prepare_training():
    file_path = "training_data_jira.json"

    if not Path(file_path).exists():
        return {"error": f"{file_path} not found."}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {"error": "training_data_jira.json is empty!"}

    sft_data = [{"instruction": item["input"], "output": item["output"]} for item in data]

    with open("sft_training_data.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)

    return {
        "message": "SFT training data prepared.",
        "total_items": len(sft_data),
        "sft_training_file": "sft_training_data.json"
    }

 
@app.post("/reset")
async def reset(request: Request):
    
    data =  await request.json()
    print(data)
    
    file_list = [
    "training_data_jira.json",
    "sft_training_data.json"
]

    for file in file_list:
        with open(file, 'w', encoding='utf-8'):
            pass
    
    return data

@app.post("/sft_train")
async def sft_train(
    base_model_path: Optional[str] = "distilgpt2",
    sft_data_file: Optional[str] = "sft_training_data.json",
    output_dir: Optional[str] = "slm-finetuned",
    load_in_8bit: Optional[bool] = False
):
    """
    Run Supervised Fine-Tuning (SFT) on a base model using instruction-response data.
    """
    try:
        import json
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

        print("[INFO] Loading SFT training data...")
        with open(sft_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[INFO] Loaded {len(data)} training samples.")
        print("[INFO] Formatting data for instruction-tuning...")
        dataset = Dataset.from_list([
            {"text": f"### Instruction:\n{item['instruction']}\n### Response:\n{item['output']}"}
            for item in data
        ])

        print(f"[INFO] Loading tokenizer from '{base_model_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[INFO] Loading model from '{base_model_path}'...")
        model_kwargs = {}
        if load_in_8bit:
            print("[INFO] Enabling 8-bit quantization...")
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

        print("[INFO] Tokenizing dataset...")
        def preprocess(example):
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()  # ✅ Add labels!
            return tokenized
        tokenized = dataset.map(preprocess)
        print("[INFO] Setting training arguments...")
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            save_steps=100,
            logging_steps=20,
            learning_rate=2e-5,
            fp16=False
        )

        print("[INFO] Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
        )

        print("[INFO] Starting training...")
        trainer.train()

        print(f"[INFO] Saving model and tokenizer to '{output_dir}'...")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)

        print("[SUCCESS] SFT training completed successfully.")
        return {
            "message": "SFT training complete.",
            "output_dir": output_dir
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"error": str(e)}


@app.post("/embed_store")
async def embed_store():
    file_path = "training_data_jira.json"
    if not Path(file_path).exists():
        return {"error": f"{file_path} not found."}

    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        return {"error": "No items to embed."}

    print(f"🔍 Found {len(items)} issues to embed into Qdrant...")

    vector_size = embed_model.get_sentence_embedding_dimension()
    setup_collection(vector_size)

    points = []
    for idx, item in enumerate(items):
        vector = embed_model.encode(item["input"])
        point_id = str(uuid.uuid4())
        payload = item["meta"]
        payload.update({"input": item["input"], "output": item["output"]})

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)

    return {
        "message": f"✅ Successfully embedded and stored {len(points)} items to Qdrant.",
        "collection": collection_name
    }

def setup_collection(vector_size: int):
    existing_collections = client.get_collections().collections
    for coll in existing_collections:
        client.delete_collection(collection_name=coll.name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    file_path = "training_data_jira.json"
    if not Path(file_path).exists():
        return {"error": f"{file_path} not found."}

    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        return {"error": "No items to embed."}

    print(f"🔍 Found {len(items)} issues to embed into Qdrant...")

    # Step 1: Setup Qdrant collection (only once)
    vector_size = embed_model.get_sentence_embedding_dimension()
    setup_collection(vector_size)

    points = []
    for idx, item in enumerate(items):
        vector = embed_model.encode(item["input"])
        point_id = str(uuid.uuid4())
        payload = {
            "key": item["meta"]["key"],
            "status": item["meta"]["status"],
            "project": item["meta"]["project"],
            "priority": item["meta"]["priority"],
            "type": item["meta"]["type"],
            "input": item["input"],
            "output": item["output"]
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)

    return {
        "message": f"✅ Successfully embedded and stored {len(points)} items to Qdrant.",
        "collection": collection_name
    }
    

@app.post("/rag_answer")
async def rag_answer(query_input: QueryInput):
    query = query_input.query
    print(f"💬 Received query: {query}")

    retriever = LCQdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embedding_function
).as_retriever(search_kwargs={"k": 5})

    # Sanitize retrieved documents (remove any with None or invalid content)
    def safe_get_relevant_documents(query):
        docs = retriever.get_relevant_documents(query)
        return [doc for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

    retriever.get_relevant_documents = safe_get_relevant_documents


    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="moondream"),
        chain_type="stuff",
        retriever=retriever
    )

    try:
        result = qa_chain.run(query)
        return {"answer": result.strip()}
    except Exception as e:
        return {"error": f"❌ Failed to get response from Ollama: {str(e)}"}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embed_server:app", host="0.0.0.0", port=8000, reload=False)