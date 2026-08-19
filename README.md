#  CascadeRank: Offline AI Recruitment Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Polars](https://img.shields.io/badge/Polars-Blazing_Fast-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Offline](https://img.shields.io/badge/Environment-100%25_Offline-lightgrey.svg)

**CascadeRank** is a highly optimized, strictly offline AI recruitment engine built to process massive candidate datasets under extreme hardware and time constraints.

Designed to filter **100,000 raw candidate profiles** down to the **Top 100 best matches** for a given Job Description, CascadeRank relies on a multi-stage funnel architecture that prioritizes engineering pragmatism, memory management, and deterministic speed before executing localized AI inference.

---

##  System Architecture

To survive strict 5-minute execution limits and a 16 GB RAM ceiling on a CPU-only environment, the pipeline is divided into three asynchronous processing stages, capped off with a real-time generative AI dashboard.

### Stage 1: Deterministic Filtering (`src/parser.py`)
* **Technology:** `Polars` (vectorized DataFrames).
* **Logic:** Instantly drops ghost candidates (inactive > 6 months), candidates in the wrong geographic locations, and those lacking core ML/AI domain skills.
* **Scale:** Filters 100,000 candidates down to ~12,700 in ~30 seconds.

### Stage 2: Dense Semantic Retrieval (`src/retrieval.py`)
* **Technology:** `sentence-transformers` (`all-MiniLM-L6-v2`) & `faiss-cpu`.
* **Logic:** Generates text embeddings for the surviving candidate profiles and builds an in-memory vector index. Performs cosine similarity search against the embedded Job Description.
* **Optimization:** Forces aggressive garbage collection (`gc.collect()`) immediately post-retrieval to prevent memory leaks. Extracts the Top 300 candidates.

### Stage 3: Behavioral Re-Ranking (`src/ranker.py`)
* **Technology:** Pure Python heuristic logic.
* **Logic:** Applies percentage multipliers to the Stage 2 semantic score based on positive/negative behavioral signals (e.g., short notice periods, high GitHub activity).
* **Output:** Ranks the final Top 100 candidates and exports a clean submission CSV.

### The Dashboard & Real-Time AI Inference (`app.py`)
* **Technology:** `Streamlit`, `llama-cpp-python`, and `Phi-3-Mini-4K-Instruct` (GGUF).
* **Logic:** Instead of pre-computing expensive LLM reasoning for all candidates — which breaks the time constraints and hardware limits — the dashboard uses **lazy loading**. When a judge selects a candidate, the app dynamically loads a quantized LLM into RAM, applies strict context whitelisting to prevent token degradation, and generates a live, offline justification for the candidate match.

---

##  Prerequisites: Offline Model Setup (Required)

To ensure strict data privacy and adhere to the 100% offline constraint, CascadeRank runs all inference locally on your CPU. To keep this repository lightweight, the large model binaries are excluded via `.gitignore` — pushing gigabytes of model weights to GitHub bloats the repo, slows cloning to a crawl, and typically fails outright since GitHub blocks files over 100 MB.

Before running the pipeline, you **must manually download** the following models and place them in the `models/` directory at the root of the project.

### 1. Create the directory

```bash
mkdir models
```

### 2. Download the Semantic Retrieval Model (`all-MiniLM-L6-v2`)

Handles Stage 2 (Dense Semantic Search).

* **Download:** [Insert your source link — HuggingFace repo or prepared Google Drive ZIP]
* **Placement:** Extract the contents so the folder sits exactly at `models/all-MiniLM-L6-v2/`.

### 3. Download the Reasoning LLM (`Phi-3-Mini 4K GGUF`)

Handles the live AI candidate reasoning in the Streamlit dashboard.

* **Download:** [Insert link to `Phi-3-mini-4k-instruct-q4.gguf`]
* **Placement & Renaming:** Place the downloaded `.gguf` file inside `models/`. **You must rename it** to exactly `phi-3-mini.gguf` so the pipeline can locate it.

> 💡 **Tip:** Rather than sending judges to raw HuggingFace pages to hunt for the right files, zip both the `all-MiniLM-L6-v2` folder and the `phi-3-mini.gguf` file together into a single `CascadeRank_Models.zip`, upload it to Google Drive, and share that one link above. It turns setup from a multi-step scavenger hunt into a single download.

###  Final directory check

Before running `python -m src.pipeline`, your file tree should look exactly like this:

```text
cascade-rank/
│
├── models/
│   ├── all-MiniLM-L6-v2/     <-- Folder containing config.json, pytorch_model.bin, etc.
│   └── phi-3-mini.gguf       <-- The renamed Phi-3 model file
│
├── src/
├── app.py
└── README.md
```

---

##  Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/cascade-rank.git
cd cascade-rank
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** If you hit Streamlit watcher errors related to image processing, make sure Torchvision is installed in CPU-only mode:
> ```bash
> pip install torchvision --extra-index-url https://download.pytorch.org/whl/cpu
> ```

---

##  Usage

### 1. Execute the pipeline

Run the master orchestrator to parse, embed, and rank the candidates. This runs entirely offline and generates `data/output/final_submission.csv` in roughly 7–8 minutes.

```bash
python -m src.pipeline
```

### 2. Launch the AI dashboard

Once the pipeline finishes, boot up the Streamlit interface to view the Top 100 list and test the real-time Phi-3 reasoning engine.

```bash
streamlit run app.py
```

---


