# Wikipedia Q&A Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about a knowledge base of 50,000 Wikipedia articles. It uses a local ChromaDB vector store and Google's Gemini API to provide answers grounded in the document sources, preventing hallucination.

---

## Features
* **Data Pipeline:** Downloads and processes 50,000 random articles from the Wikipedia dataset.
* **Semantic Search:** Uses the `sentence-transformers/all-MiniLM-L6-v2` model to create text embeddings and stores them in a ChromaDB vector store.
* **Grounded Generation:** Generates answers using Google's Gemini API (`gemini-2.5-flash-lite`) based on the retrieved document chunks.
* **Reliability Guardrails:** Implements both prompt-based and code-based guardrails to ensure answers are based only on the provided sources.
* **Interactive UI:** A simple web interface built with Streamlit.
* **Deployment:** Fully containerized with Docker for easy and reproducible deployment.

---

## Tech Stack
* **Backend:** Python
* **AI Framework:** LangChain
* **UI:** Streamlit
* **Vector Store:** ChromaDB
* **Embedding Model:** Hugging Face `sentence-transformers`
* **LLM:** Google Gemini API (`gemini-2.5-flash-lite`)
* **Deployment:** Docker

---

## Local Setup
Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
```

### 2. Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

Create a `.env` file in the project's root directory and add your Google AI Studio API key.

```bash
# .env
GOOGLE_API_KEY='YOUR_API_KEY_HERE'
```

### 5. Build the Knowledge Base

Run the data pipeline scripts in order.  
*Note: This process can be time-consuming depending on the number of articles.*

```bash
# 1. Download and save the raw Wikipedia articles
python 1_curate_data.py

# 2. Chunk, embed, and store the articles in the database
python 2_ingest.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

---

## Running with Docker

The project can also be run using Docker, which handles all dependencies and setup automatically.  
Ensure your local `data/chroma_db` is built first by running the scripts in Step 5.

### 1. Build the Docker Image

```bash
docker build -t rag-wikipedia .
```

### 2. Run the Docker Container

Run the container and pass your API key as an environment variable.

```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY="YOUR_API_KEY_HERE" rag-wikipedia
```

Access the application in your browser at:  
[http://localhost:8501](http://localhost:8501)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
