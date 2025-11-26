#app.py
import boto3, subprocess, json
from typing import List
from fastapi import FastAPI
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
class TextInput(BaseModel):
    input: List[List[str]]

def call_worker(model_path, payload):
    proc = subprocess.run(
        ["python3", "worker.py", model_path],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed: {proc.stderr}")
    return json.loads(proc.stdout)

# Application endpoint
@app.post("/embed")
def get_embedding(data: TextInput):
    input = data.input
    # Run the language sorting worker leveraging the fasttext language detection model
    # Returns a dictionnary having one key for each language FR & EN 
    # For each key, we have two lists
    # The first is the index of that language job field in the input list
    # The second is a list for each job field containing the list of tokens of this job's field
    print("Calling worker for language identification")
    try:
        group_input=call_worker(LOCAL_LANG_MODEL_PATH, input)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on language identification: {e}")
    print("Retrieving worker output")

    # With the output grouped by language key we can run the inference in batches
    # The inference is applied running a subprocess on the second list
    print("Calling worker for french model inference")
    FR_input = group_input["FR"][1]
    try:
        FR_output = call_worker(LOCAL_FR_MODEL_PATH,FR_input)["embeddings"]
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on french model inference: {e}")
    print("French model inference retrieved")

    print("Calling worker for english model inference")
    EN_input = group_input["EN"][1]
    try:
        EN_output = call_worker(LOCAL_EN_MODEL_PATH,EN_input)["embeddings"]
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"worker failed on english model inference: {e}")
    print("English model inference retrieved")

    # Now that we got the embeddings for the french and english key
    # We can reorder the embeddings to match the original order of job's fields input
    
    # We join the french and english indexes
    order=group_input["FR"][0]+group_input["EN"][0]
    # Make sure that the index list is complete
    assert(sorted(order) == list(range(len(input))))
    # We join the outputs in the same order
    output=FR_output+EN_output
    # Get the embeddings in their original order
    ordered_output=[output[i] for i in order]

    return (ordered_output)

@app.get("/health")
def health():
    return {"status": "ok"}