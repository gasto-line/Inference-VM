# worker.py
import sys, json, fasttext
model_path = sys.argv[1]
model = fasttext.load_model(model_path)
payload = json.load(sys.stdin)

# payload contains list of texts / tokens ...
if "lang" in model_path:
    result = {"FR": [[],[]], "EN": [[],[]]}
    for i,text in enumerate(payload):
        lang = model.predict(text)[0][0]
        if lang == "__label__fra_Latn":
            result["FR"][0].append(i)
            result["FR"][1].append(text)
        elif lang == "__label__eng_Latn":
            result["EN"][0].append(i)
            result["EN"][1].append(text)
else:
    result = {"embeddings": []}
    for text in payload:
        result["embeddings"].append(model.get_sentence_vector(text).tolist())

print(json.dumps(result))
# process exits -> memory freed