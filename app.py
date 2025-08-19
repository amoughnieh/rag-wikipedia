import streamlit as st


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

st.title("AI-Powered Wikipedia Explorer")

@st.cache_resource
def load_chain():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'data')

    persist_directory = os.path.join(db_path, 'chroma_db')

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(db._collection.metadata)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    template = '''
    Answer the question based only on the following knowledge base:
    {context}
    
    Question: {input}
    
    Please remember, if the knowledge base does not include relevant information
    pertaining to the question, do not provide information from your own
    memory, only provide information from the given knowledge base.
    '''
    prompt = ChatPromptTemplate.from_template(template)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.3,
                       'k': 6}
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

chain = load_chain()

user_question = st.text_input("Ask a question about the articles:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_question})

        if not response["context"]:
            st.header("Answer")
            st.write("I'm sorry, I couldn't find any relevant information in the documents to answer your question.")
            with st.expander("Show Sources"):
                st.write("Number of documents: 0")
        else:
            st.header("Answer")
            st.write(response["answer"])

            with st.expander("Show Sources"):
                for doc in response["context"]:
                    st.write(f"**Source:** {doc.metadata.get('title', 'Unknown Title')}, **ID:** {doc.metadata.get('id', 'Unknown ID')}")
                    st.write(f"**URL:** {doc.metadata.get('url', 'No URL')}")
                    st.write(f"**Content:** {doc.page_content}")
                    st.write("---")
                st.write(f"Number of documents: {len(response['context'])}")
    else:
        st.warning("Please enter a question first.")

