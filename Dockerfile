FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]