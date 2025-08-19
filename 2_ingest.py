import os
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'data', 'raw_documents')

files = os.listdir(data_folder)

db_path = os.path.join(script_dir, 'data', 'chroma_db')

if not os.path.exists(db_path):
    document_to_store = []
    for file in files:
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
            content = json_dict['text']
            metadata = {key: value for key, value in json_dict.items() if key != 'text'}
            document = Document(page_content=content,
                                metadata=metadata)
            document_to_store.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document_to_store)
    min_chunk_size = 50
    long_texts = [doc for doc in texts if len(doc.page_content) > min_chunk_size]
    print(f"Original number of chunks: {len(texts)}")
    print(f"Number of chunks after filtering: {len(long_texts)}")

    # creating vector database using filtered chunks
    print('Creating the vector database...')
    db = Chroma.from_documents(long_texts,
                               embedding_function,
                               persist_directory=db_path)

    print('Finished creating the vector database.')

else:
    print('Vector database already exists. Loading...')
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function
)
    print('Vector database loaded')

print("Checking titles in the database...")

retrieved_items = db.get(
    limit=1000000,
    include=['metadatas']
)

unique_titles = set()
for metadata in retrieved_items['metadatas']:
    if 'title' in metadata:
        unique_titles.add((metadata['title'], metadata['id']))

print(f"\n--- {len(unique_titles)} Unique Article Titles Found ---")
for title in sorted(list(unique_titles)):
    print(title)

if __name__ == '__main__':
    pass