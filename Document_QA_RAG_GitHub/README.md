# AI-Powered Document Question Answering System (RAG)

A starter project that lets a user upload a PDF, indexes its text with Hugging Face
sentence-transformer embeddings and FAISS, retrieves relevant chunks, and asks an
OpenAI chat model to answer using only the retrieved context.

## Architecture

1. Streamlit provides the PDF upload and chat interface.
2. Streamlit sends the PDF and questions to the FastAPI service.
3. `pypdf` extracts page text; `RecursiveCharacterTextSplitter` creates overlapping chunks.
4. `HuggingFaceEmbeddings` converts chunks into vectors.
5. FAISS performs semantic similarity retrieval.
6. LangChain builds the prompt and calls the OpenAI API.
7. The API returns the answer and page-based source excerpts.

## Requirements

- Python 3.10 or 3.11 recommended.
- An OpenAI API key with API access and billing configured.
- Internet access on first run to download the Hugging Face embedding model.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your_openai_api_key_here` with your key. Never commit `.env`
or expose the key in frontend code.

## Run the application

Open **Terminal 1** in the project directory:

```powershell
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Open **Terminal 2**:

```powershell
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually `http://localhost:8501`).
Upload a text-based PDF, click **Process PDF**, and ask a question. API docs are at
`http://127.0.0.1:8000/docs`; health check is `http://127.0.0.1:8000/health`.

## Run on macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, activate the environment and run:

```bash
streamlit run streamlit_app.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/documents/upload` | Multipart upload (`session_id`, `file`) and index PDF |
| POST | `/ask` | JSON body: `{"session_id":"...","question":"..."}` |
| DELETE | `/documents/{session_id}` | Remove the in-memory index |

## Configuration

See `.env.example`. `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` control chunking and retrieval.
The embedding model runs locally on CPU by default. The first startup may take longer while
the model is downloaded.

## Important limitations / production improvements

- **In-memory index:** document vectors are held in the API process memory. They disappear
  when the API restarts and are not shared across multiple workers. Persist FAISS indexes
  with safe ownership/session controls or use a managed vector database for production.
- **Scanned PDFs:** this version extracts embedded text only. Add OCR (for example Tesseract)
  for image-only PDFs.
- **Security:** this is a learning/demo baseline. Add authentication, authorization, rate
  limiting, per-user quotas, malware scanning, HTTPS, and secure document deletion before
  public deployment. Do not expose the API publicly without these controls.
- **OpenAI cost:** each question makes an API request. Monitor usage and configure limits.
- **Grounding:** retrieval and prompts reduce unsupported answers but cannot guarantee
  perfect factuality. Verify important answers against the cited source pages.

## Troubleshooting

- `OPENAI_API_KEY is missing`: ensure `.env` exists in the project root and contains a valid key.
- OpenAI authentication/billing errors: verify the key and API account access.
- `No extractable text found`: PDF may be scanned or image-only; OCR is required.
- Streamlit cannot connect: start FastAPI first and confirm the sidebar API URL.
- Model download errors: check internet access and Hugging Face connectivity.

## Project structure

```text
rag_document_qa/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   └── rag.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Tests
Run `python -m pytest` from the repository root. The smoke test checks file presence; live API calls require your own credentials.
