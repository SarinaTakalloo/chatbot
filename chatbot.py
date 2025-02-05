import streamlit as st  
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain_community.embeddings.openai import OpenAIEmbeddings as oai  # the embeddings using openai key
from langchain_community.vectorstores import FAISS #something like a database 
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI





OPENAI_API_KEY = "YOUR API KEY"



# Upload pdf files
st.header("My First Chatbot, Sarina Takalloo")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")


# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

        
    # Break it into chunks
    text_splitter = rcts(separators="\n", chunk_size = 1000, chunk_overlap = 150, length_function = len)
    chunk = text_splitter.split_text(text)
    # st.write(chunk)


    # Generating embeddings
    embeddings = oai(openai_api_key = OPENAI_API_KEY)    


    # # Creaing vectors faiss
    vector_store = FAISS.from_texts(chunk, embeddings)


    # Get user question
    user_question = st.text_input("Type your question in here")

    # Do similarity check
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)


        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )


        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)

