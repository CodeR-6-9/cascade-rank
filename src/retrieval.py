import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from docx import Document
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
MODEL_PATH = BASE_DIR.parent / "models" / "all-MiniLM-L6-v2"
DATA_PATH = BASE_DIR.parent / "data" / "raw" / "candidates.jsonl"
JD_PATH = BASE_DIR.parent / "data" / "raw" / "job_description.docx"
EXPECTED_DIMENSIONS = 384  

def build_retrieval_document(candidate):
    profile = candidate.get("profile", {})

    skill_lines = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "Unknown").title()
        skill_lines.append(f"- {name} ({proficiency})")
    skills_text = "\n".join(skill_lines)

    career_entries = []
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        company = job.get("company", "")
        description = job.get("description", "")
        career_entries.append(f"{title} | {company}\n{description}")
    career_text = "\n\n".join(career_entries)

    education_entries = []
    for edu in candidate.get("education", []):
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        education_entries.append(f"{degree} in {field}")
    education_text = "\n".join(education_entries)

    sections = [
        "Current Title:", profile.get("current_title", ""), "",
        "Years of Experience:", str(profile.get("years_of_experience", "")), "",
        "Headline:", profile.get("headline", ""), "",
        "Professional Summary:", profile.get("summary", ""), "",
        "Skills:", skills_text, "",
        "Career History:", "", career_text, "",
        "Education:", education_text,
    ]

    return "\n".join(sections).strip()

def validate_embeddings(embeddings, expected_count, expected_dimensions):
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")

    actual_count = embeddings.shape[0]
    actual_dimensions = embeddings.shape[1]
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()

    if actual_count != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {actual_count}")
    if actual_dimensions != expected_dimensions:
        raise ValueError(f"Expected embedding dimension {expected_dimensions}, got {actual_dimensions}")
    if has_nan:
        raise ValueError("Embeddings contain NaN values")
    if has_inf:
        raise ValueError("Embeddings contain Inf values")

    return {
        "shape": embeddings.shape,
        "count": actual_count,
        "dimensions": actual_dimensions,
        "dtype": embeddings.dtype,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

def validate_artifacts(loaded_index, mapping, metadata, expected_dimensions):
    if loaded_index.d != expected_dimensions:
        raise ValueError(f"Expected index dimension {expected_dimensions}, got {loaded_index.d}")
    if loaded_index.ntotal != len(mapping):
        raise ValueError(f"Index contains {loaded_index.ntotal} vectors, but mapping contains {len(mapping)} entries.")
    if len(mapping) != len(metadata):
        raise ValueError(f"Mapping size ({len(mapping)}) does not match metadata size ({len(metadata)}).")
        
def build_and_save_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]
    
    candidates = candidates[:1000]  
    
    processed_candidates = [
        {
            "candidate_id": candidate["candidate_id"],
            "text": build_retrieval_document(candidate)
        }
        for candidate in candidates
    ]
    
    texts = [c["text"] for c in processed_candidates]
    model = SentenceTransformer(str(MODEL_PATH))
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype('float32')
    
    info = validate_embeddings(embeddings, len(processed_candidates), EXPECTED_DIMENSIONS)
    print(info)
    
    index = faiss.IndexFlatIP(EXPECTED_DIMENSIONS)
    index.add(embeddings)
    
    candidate_mapping = {
        idx: candidate["candidate_id"]
        for idx, candidate in enumerate(processed_candidates)
    }
    
    candidate_metadata = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(ARTIFACTS_DIR / "candidate_index.faiss"))
    
    with (ARTIFACTS_DIR / "candidate_mapping.pkl").open("wb") as f:
        pickle.dump(candidate_mapping, f)
        
    with (ARTIFACTS_DIR / "candidate_metadata.pkl").open("wb") as f:
        pickle.dump(candidate_metadata, f)

def load_artifacts():
    index = faiss.read_index(str(ARTIFACTS_DIR / "candidate_index.faiss"))
    
    with (ARTIFACTS_DIR / "candidate_mapping.pkl").open("rb") as f:
        candidate_mapping = pickle.load(f)
        
    with (ARTIFACTS_DIR / "candidate_metadata.pkl").open("rb") as f:
        candidate_metadata = pickle.load(f)
        
    validate_artifacts(index, candidate_mapping, candidate_metadata, EXPECTED_DIMENSIONS)
    
    return index, candidate_mapping, candidate_metadata

def load_job_description(path):
    document = Document(path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)

def search_candidates(job_description, model, index, candidate_mapping, candidate_metadata, top_k=500):
    top_k = min(top_k, index.ntotal)
    
    query_embedding = model.encode(
        [job_description],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    query_embedding = query_embedding.astype('float32')
    
    if query_embedding.shape != (1, EXPECTED_DIMENSIONS):
        raise ValueError(f"Query embedding shape mismatch. Expected (1, {EXPECTED_DIMENSIONS}), got {query_embedding.shape}")
    if np.isnan(query_embedding).any() or np.isinf(query_embedding).any():
        raise ValueError("Query embedding contains NaN or Inf values.")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, faiss_idx in enumerate(indices[0]):
        if faiss_idx == -1:
            continue

        candidate_id = candidate_mapping[faiss_idx]
        candidate_data = candidate_metadata[candidate_id]
        score = float(distances[0][i])

        results.append({
            "candidate_id": candidate_id,
            "similarity_score": score,
            "candidate": candidate_data
        })

    return results

if __name__ == "__main__":
    build_and_save_index()
    
    index, mapping, metadata = load_artifacts()

    search_model = SentenceTransformer(str(MODEL_PATH))

    jd = load_job_description(JD_PATH)

    top_candidates = search_candidates(jd, search_model, index, mapping, metadata, top_k=500)