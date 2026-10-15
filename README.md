# AI-Assisted-Assembly-Error-and-Retrieval-System

Offline NLP-based defect logging and instant semantic retrieval system for aerospace assembly lines, built with Python, Sentence-Transformers, SQLite, and Streamlit.

The objective is to replace friction-heavy, paper-and-spreadsheet defect logging with an institutional knowledge loop. The system uses localized deep learning models to log assembly snags, evaluate natural language descriptions, and instantly retrieve verified past corrective actions.

REPOSITORY STRUCTURE -

ALH-Defect-Management/
│
├── Database/
│   ├── alh_errors.db         # SQLite relational database for logs & images
│   └── vectors.pkl           # Cached Sentence-BERT vector embeddings
│
├── Requirements/
│   └── models/               # Offline transformer model files
│       └── all-MiniLM-L6-v2/
│
├── app.py                    # Twin-Tab Streamlit UI & export pipeline
├── nlp_engine.py             # Vector generation and cosine similarity engine
└── requirements.txt          # Python runtime dependencies

SUBSYSTEM -

NLP & Vector Engine (nlp_engine.py)

- Responsible for:Sentence-Transformers Integration (all-MiniLM-L6-v2)   
- 84-dimensional dense text embedding generation   
- Real-time cosine similarity matrix calculation   
- Vector cache creation and serialization via pickle

Data & Storage Layer (Database/)

- Responsible for:SQLite local database schema & relational storage
- Pre-populated seed datasets for helicopter assembly snags[cite: 1]
- Duplicate detection and record deletion logic
- Storage and retrieval of snag reference photographs

Data & Storage Layer (Database/)

- Responsible for:SQLite local database schema & relational storage
- Pre-populated seed datasets for helicopter assembly snags[cite: 1]
- Duplicate detection and record deletion logic
- Storage and retrieval of snag reference photographs

  Getting Started -

  New contributors and developers should review the system setup requirements below before running the application locally.

  Prerequisites -
  
  - Python 3.10+ runtime environment
  - Standard shop-floor hardware (runs fully on CPU without cloud connectivity)
  
  Setup Order -

    1. Clone the repository to your local machine.
    2. Install dependencies:

       pip install -r requirements.txt

  3. Ensure model files are located in Requirements/models/all-MiniLM-L6-v2/

  Development Workflow -
  
  - main contains the stable, verified prototype code[cite: 1].
  - Database/alh_errors.db tracks live test logs and historical assembly records.
  - When new records are inserted via the UI, vector caches (Database/vectors.pkl) update automatically.
  - All processing remains 100% offline to maintain compliance with aerospace security standards.

  Benchmark & Operational Log (Latest Release) -

 Recent Updates -

The prototype successfully completed verification tests against known historical snags and unrecorded unique entries, reducing manual lookup times from ~40 minutes down to seconds.

System Log & Optimization Summary -

1. Semantic Retrieval: Replaced rigid keyword queries with vector similarity, matching technical synonyms (e.g., "connector loose" ↔ "plug unmated").
2. Latency Optimization: Implemented embedding caching via pickle, bypassing full dataset re-encoding during searches to achieve millisecond response times.
3. Excel Audit Pipeline: Built an openpyxl drawing generator to export live relational data along with formatted, embedded photo thumbnails for QA managers.
4. System Demonstration: Verified multi-tab navigation, network status checking, duplicate cleanup, and instant similarity scoring.

How to Run the System -

1. Launch the application interface:

streamlit run app.py

2. Test the NLP engine directly via CLI:

python nlp_engine.py

3. Log a Defect:

- Navigate to LOG Error tab.
- Fill in Component Ref, Error Type, Description, and Corrective Action.
- Click Save Error to update the database and rebuild vector caches.

4. Search Past Defect Logs:

- Navigate to SEARCH Errors tab.
- Type a plain-language fault query (e.g., "bolt not tightened enough on gearbox").
- View top 3 contextually matching records and past engineer fixes.

