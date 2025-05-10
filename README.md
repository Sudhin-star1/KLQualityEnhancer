# 🧠 Knowledge Library Quality Enhancer

An intelligent, agentic system to clean and curate large-scale QnA knowledge bases. This tool helps detect **duplicate**, **outdated**, and **redundant** entries and suggests smart actions like **merge**, **review**, or **archive** through a fully local, interactive Streamlit interface.

> Designed to scale to 50K+ entries, optimized for CPU, and ideal for secure or offline environments.

---

## 🚀 Features

- 🔍 **Semantic Duplicate Detection** using SentenceTransformer embeddings + HDBSCAN
- ⏳ **Outdated Content Detection** with rule-based heuristics using metadata
- 🤖 **Auto-suggestions** for merge, review, or archive actions
- 📊 **Streamlit Curator Dashboard** with comparison, filtering, search, and action logging
- ⚡ **Runs fully offline** (no GPU, no external APIs)

---

## 🎯 Why These Algorithms?

### ✅ SentenceTransformer Embeddings
- Captures semantic similarity across reworded QnAs
- Chosen model: `all-MiniLM-L6-v2` for CPU efficiency

### ✅ HDBSCAN Clustering
- Handles variable-density clusters
- No need to specify `k` (number of clusters)
- Isolates outliers and noise naturally

### ✅ Rule-Based Outdated Detection
- Leverages `created_at`, `deleted_at`, and `product_status`
- Works without labeled data or external dependencies

---

## 🧱 Architecture

📁 knowledge-library-quality-enhancer/
├── data/ # Raw and processed QnA CSVs
├── compute_embedding.py # Generate & cache embeddings
├── cluster.py # HDBSCAN clustering logic
├── outdated.py # Detect outdated entries
├── suggest.py # Suggest merge/review/archive actions
├── run_pipeline.py # End-to-end pipeline orchestration
├── ui.py # Streamlit-based curation dashboard
├── notebooks/ # Dev notebooks and visualizations
└── README.md


---

## 🖥️ Streamlit UI Highlights

- Visualize QnA clusters
- Compare entries side-by-side
- Accept/reject action suggestions
- Filter by category, product, or suggestion type
- Log curator actions to CSV

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/knowledge-library-quality-enhancer.git
cd knowledge-library-quality-enhancer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run ui.py
```

## 🧪 Sample Workflow
Upload your QnA CSV with fields like: question, answer, details, created_at, deleted_at, product_id, category

Run the full pipeline via run_pipeline.py or UI

Review suggestions in the dashboard

Export final decisions to update your knowledge base

## 📝 Contribution
Contributions are welcome! If you have ideas for:

Custom LLM-based suggestion refinement

Real-time feedback loops

Fine-tuned domain embeddings

Open an issue or create a pull request 🚀

## 📄 License
MIT License. See LICENSE for details.

## 🙌 Acknowledgments
Built for the SecurityPal Hackathon as a demonstration of agentic systems and scalable knowledge engineering.

