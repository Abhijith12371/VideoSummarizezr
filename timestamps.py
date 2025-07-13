import whisper
import torch
from sentence_transformers import SentenceTransformer, util
import json

# ----- Config -----
VIDEO_PATH = "video.mp4"
EMBED_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.65  # Lower = more sensitive to change

# ----- Load Models -----
print("🧠 Loading Whisper model...")
asr_model = whisper.load_model("base")

print("🔍 Transcribing video...")
result = asr_model.transcribe(VIDEO_PATH, verbose=False)

segments = result["segments"]
texts = [seg["text"] for seg in segments]

print("🔎 Loading SentenceTransformer...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("💬 Embedding segments...")
embeddings = embed_model.encode(texts, convert_to_tensor=True)

# ----- Compute Similarity and Detect Changes -----
print("🔁 Computing similarity between adjacent segments...")
timestamps = []
for i in range(len(embeddings)):
    if i == 0:
        timestamps.append({
            "time": f"{int(segments[i]['start'] // 60):02d}:{int(segments[i]['start'] % 60):02d}",
            "label": segments[i]['text'].strip().split('.')[0][:60] + "..."
        })
    else:
        sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()
        if sim < SIMILARITY_THRESHOLD:
            ts = segments[i]['start']
            label = segments[i]['text'].strip().split('.')[0][:60]
            timestamps.append({
                "time": f"{int(ts // 60):02d}:{int(ts % 60):02d}",
                "label": label + "..."
            })

# ----- Output Results -----
print("\n📍 Generated Timestamps:\n")
for ts in timestamps:
    print(f"[{ts['time']}] {ts['label']}")

with open("semantic_timestamps.json", "w") as f:
    json.dump(timestamps, f, indent=2)

print("\n✅ Saved timestamps to semantic_timestamps.json")
