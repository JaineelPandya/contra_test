# RegGraph

**RegGraph** is a backend-only engine that transforms complex regulatory and legal PDF documents (such as NSE, SEBI, and RBI circulars) into a **hierarchical, temporal knowledge graph stored in Neo4j**.

---

## System Architecture

```text
PDF Upload
    ↓
Metadata Extraction (Regex + LLM Fallback)
    ↓
Text Extraction (PyMuPDF)
    ↓
Hierarchy Detection (Sections, Subsections, Clauses, Points, FAQs)
    ↓
Hierarchical Chunking (Path Provenance & NEXT/PREVIOUS Links)
    ↓
Atomic Fact Extraction (14 Predicate Categories)
    ↓
Entity Extraction & Resolution (Canonical Normalization & Aliases)
    ↓
Cross-Document Reference Detection (Citations & Target Clause Resolution)
    ↓
Change Detection & Temporal Tracking (DELETED, MODIFIED, SUPERSEDED, CLARIFIED)
    ↓
Neo4j Storage (Parameterized Graph Ingestion & Provenance Wiring)
```

---

## Technology Stack

- **Language & Runtime**: Python 3.12
- **Web Framework**: FastAPI, Uvicorn
- **Validation**: Pydantic v2, Pydantic Settings
- **PDF Engine**: PyMuPDF (`fitz`)
- **Graph Database**: Neo4j 5+ (Official Python Driver)
- **LLM Abstraction**: Ollama (local) & OpenAI-compatible APIs
- **Infrastructure**: Docker, Docker Compose

---

## Getting Started

### 1. Configure Environment

Copy the template configuration:

```bash
cp .env.example .env
```

Configure your `.env` settings:

```env
# Application
APP_NAME=RegGraph
PORT=8080
HOST=0.0.0.0

# Neo4j Database
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=reggraph_password
NEO4J_DATABASE=neo4j

# LLM Provider: "groq", "gemini", "ollama", "openai", or "mock"
LLM_PROVIDER=groq

# Groq Settings (Default Active Provider - Ultra Fast)
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_TEMPERATURE=0.0

# Google Gemini Settings (if LLM_PROVIDER=gemini)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-flash-latest
GEMINI_TEMPERATURE=0.0

# Ollama Settings (if LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_TEMPERATURE=0.0

# OpenAI-Compatible API Settings (if LLM_PROVIDER=openai)
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

---

### 2. Start Services with Docker Compose

Run:

```bash
docker compose up --build
```

Services exposed:
- **FastAPI Backend**: [http://localhost:8080](http://localhost:8080)
- **Interactive Swagger Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (Username: `neo4j`, Password: `reggraph_password`)
- **Neo4j Bolt Protocol**: `bolt://localhost:7687`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload PDF and start async graph pipeline |
| `GET` | `/api/documents/{document_id}/status` | Check processing stage and extracted counts |
| `GET` | `/api/documents` | List all processed documents |
| `GET` | `/api/documents/{document_id}` | Retrieve document metadata |
| `GET` | `/api/documents/{document_id}/hierarchy` | Fetch hierarchical tree of sections/clauses |
| `GET` | `/api/documents/{document_id}/facts` | Fetch extracted atomic facts and status |
| `GET` | `/api/documents/{document_id}/changes` | Fetch detected regulatory changes and amendments |
| `GET` | `/health` | Health check for API and Neo4j connectivity |

---

## Uploading Documents via cURL

### 1. Upload Base Circular (2022 Circular with 12 Illustrative Points)

```bash
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/sample_data/circular_2022_50957.pdf"
```

Response:
```json
{
  "document_id": "84d728ea-8b43-4318-80f4-526715f212ce",
  "filename": "circular_2022_50957.pdf",
  "status": "PROCESSING"
}
```

### 2. Check Status

```bash
curl -X GET "http://localhost:8080/api/documents/84d728ea-8b43-4318-80f4-526715f212ce/status"
```

Response:
```json
{
  "document_id": "84d728ea-8b43-4318-80f4-526715f212ce",
  "status": "COMPLETED",
  "current_stage": "COMPLETED",
  "pages_processed": 2,
  "total_pages": 2,
  "sections_created": 14,
  "chunks_created": 14,
  "facts_extracted": 31,
  "errors": []
}
```

### 3. Upload Amending Circular (2026 Circular Modifying P1 & P2, Deleting P3-12)

```bash
curl -X POST "http://localhost:8080/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/sample_data/circular_2026_amendment.pdf"
```

---

## Neo4j Cypher Query Reference

Connect to Neo4j Browser at [http://localhost:7474](http://localhost:7474) and execute the following queries:

### 1. View All Documents
```cypher
MATCH (d:Document)
RETURN d;
```

### 2. View Document Hierarchy
```cypher
MATCH p = (d:Document)-[:HAS_SECTION|HAS_SUBSECTION|HAS_CLAUSE*1..4]->(n)
RETURN p;
```

### 3. View Sequential Chunks (NEXT / PREVIOUS)
```cypher
MATCH p = (c1:Chunk)-[:NEXT]->(c2:Chunk)
RETURN p;
```

### 4. View Facts with Complete Provenance
```cypher
MATCH (f:Fact)-[:SUPPORTED_BY]->(c:Chunk)<-[:HAS_CHUNK]-(clause)
RETURN f, c, clause;
```

### 5. View Cross-Document Reference & Amendment Links
```cypher
MATCH p = (new_doc:Document)-[:REFERENCES|AMENDS]->(old_doc:Document)
RETURN p;
```

### 6. View Deleted Facts and Their Temporal Lifecycle
```cypher
MATCH (f:Fact)
WHERE f.status = "DELETED"
RETURN f.subject, f.predicate, f.object, f.status, f.valid_from, f.valid_to, f.source_text;
```

### 7. View Fact Evolution (MODIFIES, DELETES, SUPERSEDES, CLARIFIES)
```cypher
MATCH p = (new_fact:Fact)-[:MODIFIES|DELETES|SUPERSEDES|CLARIFIES|CONTRADICTS]->(old_fact:Fact)
RETURN p;
```

### 8. View Change Events
```cypher
MATCH (ce:ChangeEvent)
RETURN ce.change_type, ce.target_point_or_clause, ce.description, ce.effective_date
ORDER BY ce.created_at DESC;
```

### 9. View Entities and Their Facts
```cypher
MATCH (sub:Entity)<-[:SUBJECT]-(f:Fact)-[:OBJECT]->(obj:Entity)
RETURN sub.name, f.predicate, obj.name;
```

---

## Running Local Unit & Integration Tests

To run the automated test suite locally:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests with pytest
pytest -v
```

To generate synthetic test circulars:

```bash
python tests/generate_sample_pdfs.py
```
