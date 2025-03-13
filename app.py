import streamlit as st
import os
import sys
import tempfile
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        st.error("⚠️ OpenAI API key not found. Please set it in the .env file!")
        return False
    return True

def translate_text_batch(text: str, batch_size: int = 1000) -> str:
    """Translate text in batches while preserving context"""
    try:
        # Split text into sentences to preserve context
        sentences = text.split('.')
        batches = []
        current_batch = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            if current_length + len(sentence) > batch_size:
                if current_batch:
                    batches.append(' '.join(current_batch))
                current_batch = [sentence]
                current_length = len(sentence)
            else:
                current_batch.append(sentence)
                current_length += len(sentence)
        
        if current_batch:
            batches.append(' '.join(current_batch))
        
        translated_batches = []
        for batch in batches:
            translated = GoogleTranslator(source='auto', target='en').translate(batch)
            translated_batches.append(translated)
        
        return ' '.join(translated_batches)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def get_docx_text(file) -> LangchainDocument:
    """Extract text from a .docx file"""
    doc = Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return LangchainDocument(
        page_content='\n'.join(text),
        metadata={"source": file.name, "type": "docx"}
    )

def get_pdf_text(file) -> LangchainDocument:
    """Extract text from a PDF file"""
    pdf_reader = PdfReader(file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())
    return LangchainDocument(
        page_content='\n'.join(text),
        metadata={"source": file.name, "type": "pdf", "pages": len(pdf_reader.pages)}
    )

def process_documents(docs) -> List[LangchainDocument]:
    """Process multiple documents and return a list of Document objects"""
    documents = []
    for doc in docs:
        if doc.name.lower().endswith('.pdf'):
            documents.append(get_pdf_text(doc))
        elif doc.name.lower().endswith('.docx'):
            documents.append(get_docx_text(doc))
    return documents

def get_text_chunks(documents: List[LangchainDocument]) -> List[LangchainDocument]:
    """Split documents into chunks while preserving metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = []
    for doc in documents:
        # Translate content if needed
        translated_text = translate_text_batch(doc.page_content)
        doc.page_content = translated_text
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    return chunks

def get_vectorstore(text_chunks: List[LangchainDocument]) -> Chroma:
    """Create Chroma vectorstore from document chunks"""
    embeddings = OpenAIEmbeddings()
    
    # Create a temporary directory with ASCII-only path
    temp_base = os.path.join(tempfile.gettempdir(), "chromadb_" + str(uuid.uuid4())[:8])
    os.makedirs(temp_base, exist_ok=True)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=temp_base
        )
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        if os.path.exists(temp_base):
            try:
                import shutil
                shutil.rmtree(temp_base)
            except:
                pass
        raise e

def get_conversation_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Create conversation chain with custom prompt and retrieval settings"""
    llm = ChatOpenAI(
        temperature=0.0,  # Set to 0 for fully factual responses
        model_name="gpt-4"
    )
    
    # Custom prompt template
    prompt_template = """You are a helpful assistant that answers questions based solely on the provided context. 
    
    Context: {context}
    
    Current conversation:
    {chat_history}
    
    Question: {question}
    
    Instructions:
    1. Only answer with information found in the provided context
    2. If information is not in the context, say "I don't have enough information to answer that question"
    3. Keep responses clear, concise, and factual
    4. Do not mention or reference any document names or sources
    5. Do not infer or assume information that is not explicitly stated in the context.
    6. If the question requires interpretation or additional details not found in the context, state that you cannot provide an answer.
    7. If multiple relevant facts exist in the context, summarize them objectively without adding personal opinions or assumptions.
    8. Prioritize the most relevant and recent information if multiple similar details exist.
    9. Structure responses logically, ensuring coherence and readability.
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False  # Disable source tracking
    )
    
    return conversation_chain

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = []
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    try:
        st.set_page_config(page_title="Chat with Documents")
        st.header("Chat with your Documents 💬")
        
        # Add debug info section in sidebar
        with st.sidebar:
            st.subheader("Debug Information")
            st.text(f"Python version: {sys.version}")
            st.text(f"Working directory: {os.getcwd()}")
            if st.session_state.debug_info:
                for info in st.session_state.debug_info:
                    st.text(info)
            
            st.subheader("Your documents")
            uploaded_files = st.file_uploader(
                "Upload your PDFs/DOCXs here",
                accept_multiple_files=True,
                type=['pdf', 'docx']
            )

            if uploaded_files:
                st.session_state.debug_info = []  # Clear previous debug info
                
                # Process documents
                with st.spinner("Processing documents..."):
                    try:
                        documents = process_documents(uploaded_files)
                        st.session_state.debug_info.append(f"Processed {len(documents)} documents")
                        
                        text_chunks = get_text_chunks(documents)
                        st.session_state.debug_info.append(f"Created {len(text_chunks)} text chunks")
                        
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.debug_info.append("Created vector store")
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.debug_info.append("Created conversation chain")
                        
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.debug_info.append(f"Error: {str(e)}")
                        return

        # Chat interface
        if st.session_state.conversation is not None:
            user_question = st.text_input("Ask a question about your documents:")
            
            if user_question:
                with st.spinner("Thinking..."):
                    with get_openai_callback() as cb:
                        try:
                            response = st.session_state.conversation({
                                "question": user_question
                            })
                            st.write(response["answer"])
                            
                            # Display token usage
                            st.sidebar.text(f"Tokens used: {cb.total_tokens}")
                            st.sidebar.text(f"Cost: ${cb.total_cost:.4f}")
                        except Exception as e:
                            st.error(f"Error getting response: {str(e)}")
                            st.session_state.debug_info.append(f"Error: {str(e)}")
        else:
            st.info("Upload documents to start chatting!")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.session_state.debug_info.append(f"Fatal error: {str(e)}")

if __name__ == '__main__':
    main()
