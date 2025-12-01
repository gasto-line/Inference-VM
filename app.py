#app.py
import boto3, subprocess, json, os, time, psutil
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Variables ---
BUCKET="sagemaker-eu-west-3-992382721552"  
REGION="eu-west-3"
S3_FR_MODEL_KEY="facebook/fasttext-fr-vectors/model.bin"
S3_EN_MODEL_KEY="facebook/fasttext-en-vectors/model.bin"
S3_LANG_MODEL_KEY="facebook/fasttext-language-identification/model.bin"
LOCAL_FR_MODEL_PATH="/home/ec2-user/fr_model.bin"
LOCAL_EN_MODEL_PATH="/home/ec2-user/en_model.bin"
LOCAL_LANG_MODEL_PATH="/home/ec2-user/lang_model.bin"

# Intermediate paths
input_path = "/home/ec2-user/input.json"
fr_embeddings_path = "/home/ec2-user/fr_embeddings.json"
en_embeddings_path = "/home/ec2-user/en_embeddings.json"

s3 = boto3.client("s3", region_name = REGION)

# Download model from S3
print("Downloading from S3...")
# Choose the same directory as s3 to store on EBS volume
s3.download_file(BUCKET, S3_FR_MODEL_KEY, LOCAL_FR_MODEL_PATH)
s3.download_file(BUCKET, S3_EN_MODEL_KEY, LOCAL_EN_MODEL_PATH)
s3.download_file(BUCKET, S3_LANG_MODEL_KEY, LOCAL_LANG_MODEL_PATH)
print("Download completed")

app = FastAPI()

# Define input schema
class TokenInput(BaseModel):
    input: List[List[str]]

class SentenceInput(BaseModel):
    input: List[str]

def call_worker(model_path, payload, worker):
    proc = subprocess.run(
        ["python3", worker, model_path],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed: {proc.stderr}")
    return json.loads(proc.stdout)

def print_memory(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024*1024)
    print(f"[RAM] {label}: {mem_mb:.2f} MB")

def core_get_embeddings(model_type: str, input):
    # model_type ∈ {"token", "sentence"}

    worker_path= f"{model_type}_worker.py"
    if model_type not in {"token", "sentence"}:
        raise ValueError("Invalid model_type. Must be 'token' or 'sentence'.")
    
    print_memory("start")

    # Run the language sorting worker leveraging the fasttext language detection model
    # Returns a dictionnary having one key for each language FR & EN 
    # For each key, we have two lists
    # The first is the index of that language job field in the input list
    # The second is a list for each job field containing either the list of tokens of this job's field or the full sentence
    print("Calling worker for language identification")
    try:
        group_input=call_worker(model_path=LOCAL_LANG_MODEL_PATH, payload=input, worker=worker_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on language identification: {e}")
    print("Retrieving worker output")
    print_memory("after getting grouped input from language identification worker")    

    # Sanity checks
    if len(group_input["FR"][0]) + len(group_input["EN"][0]) != len(input):
        raise HTTPException(status_code=500, detail="Mismatch in number of input and grouped input items")

    # With the output grouped by language key we can run the inference in batches
    # The inference is applied running a subprocess on the second list
    print("Calling worker for french model inference")
    try:
        FR_output = call_worker(model_path=LOCAL_FR_MODEL_PATH, payload=group_input["FR"][1], worker=worker_path)["embeddings"]
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on french model inference: {e}")
    print("French model inference retrieved")
    print_memory("after getting the french model output")

    print("Calling worker for english model inference")
    try:
        EN_output = call_worker(model_path=LOCAL_EN_MODEL_PATH, payload=group_input["EN"][1], worker=worker_path)["embeddings"]
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on english model inference: {e}")
    print("English model inference retrieved")
    print_memory("after getting the english model output")

    # sanity checks
    if len(FR_output) != len(group_input["FR"][0]):
        raise HTTPException(status_code=500, detail="Mismatch in number of french embeddings")
    if len(EN_output) != len(group_input["EN"][0]):
        raise HTTPException(status_code=500, detail="Mismatch in number of english embeddings")

    
    # Create a output variable
    group_output= {
    "FR": [group_input["FR"][0], None],
    "EN": [group_input["EN"][0], None]
    }
    group_output["FR"][1]=FR_output
    group_output["EN"][1]=EN_output
    print_memory("after merging for final output")

    return (group_output)

# Application with sentence input
@app.post("/sentence")
def get_embedding(data: SentenceInput):
    input = data.input
    group_output=core_get_embeddings(model_type="sentence", input=input)
    return (group_output)

# Application with token input
@app.post("/token")
def get_embedding(data: TokenInput):
    input = data.input
    group_output=core_get_embeddings(model_type="token", input=input)
    return (group_output)

@app.get("/health")
def health():
    return {"status": "ok"}