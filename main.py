from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Get API keys from environment variables at startup
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
if not openrouter_api_key:
    print("Warning: OPENROUTER_API_KEY not found in environment variables.")

# Set up OpenRouter as the API endpoint
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = openrouter_api_key

def load_and_process_documents(pdf_path):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Get chunk parameters from environment variables or use defaults
    chunk_size = int(os.getenv("CHUNK_SIZE", 2000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 400))
    
    # Better chunking strategy with smaller chunks to avoid breaking content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # More gradual splitting
        length_function=len
    )
    
    # Create proper Document objects with enhanced metadata
    processed_docs = []
    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        chunks = text_splitter.split_text(doc.page_content)
        
        # Add more detailed metadata to each chunk
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": doc.metadata.get("source"),
                "page": page_num,
                "chunk": i,
                "content_type": "page_content"
            }
            processed_docs.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"Document processed into {len(processed_docs)} chunks")
    return processed_docs, documents

def is_toc_query(query):
    """Simple check if this is a TOC-related query"""
    toc_terms = ["table of contents", "toc", "chapters", "outline", "sections"]
    return any(term in query.lower() for term in toc_terms)

def setup_rag_chain(documents, raw_pages, model_name=None, temperature=None):
    # Get default parameters from environment variables if not provided
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-1106")
    
    if temperature is None:
        temperature = float(os.getenv("TEMPERATURE", 0.2))
    
    # Use HuggingFace embeddings 
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"Using embedding model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create vector database with Document objects
    db = FAISS.from_documents(documents, embeddings)
    
    # Enhanced prompt with better instructions for document understanding
    prompt_template = """You are an expert technical document analyst with the ability to understand complex technical information. 
    Your task is to provide accurate, thorough, and well-structured answers based on the provided context from a technical document.

    Guidelines:
    1. If the context doesn't contain the answer, clearly state this instead of guessing
    2. If answering about document structure (like table of contents), present a complete and organized view
    3. When analyzing technical content, be precise and maintain technical accuracy
    4. When asked about a specific chapter or section, focus your answer on that portion
    5. Present information in a well-structured format for readability
    6. Always provide comprehensive information - don't arbitrarily cut off listings
    7. When listing chapters or sections, ensure you include ALL of them that appear in the context

    Given the following sections from a technical document:

    {context}

    Question: {question}

    Comprehensive Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # Use ChatOpenAI with appropriate configuration
    llm = ChatOpenAI(
        model_name=model_name, 
        temperature=temperature, 
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    # Configure the chain to use the retriever and prompt
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={"k": 12}),
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs
    )

    return qa, llm, raw_pages

def run_query(qa_chain, llm, raw_pages, query):
    print(f"Processing query: {query}")
    
    # Special handling for TOC queries - use first 10 pages directly
    if is_toc_query(query):
        print("Detected TOC query - focusing on early pages")
        # Use the first 10 pages for TOC
        first_pages = raw_pages[:10]
        
        # Combine the content of the first pages
        toc_content = "\n\n".join([f"Page {i}: {page.page_content}" for i, page in enumerate(first_pages)])
        
        # Create a direct prompt for TOC extraction
        toc_prompt = f"""Extract the complete table of contents from this document, including all 
        chapters, sections, and subsections with their page numbers if available. 
        Format it in a clear, structured way.
        
        Document content:
        {toc_content}
        """
        
        # Get response directly from LLM
        response = llm.invoke(toc_prompt)
        
        # Return response with pages 0-9 as sources
        return response.content, first_pages
    
    # For regular queries, use the QA chain
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]

def main():
    parser = argparse.ArgumentParser(description="Technical Document RAG System")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--model", help="The LLM model to use (overrides .env setting)")
    parser.add_argument("--temp", type=float, help="The temperature for the LLM (overrides .env setting)")
    args = parser.parse_args()

    documents, raw_pages = load_and_process_documents(args.pdf_path)
    
    print(f"Setting up RAG chain with model: {args.model or os.getenv('MODEL_NAME', 'gpt-3.5-turbo-1106')}")
    qa_chain, llm, raw_pages = setup_rag_chain(documents, raw_pages, model_name=args.model, temperature=args.temp)
    
    print(f"Querying: {args.query}")
    answer, source_docs = run_query(qa_chain, llm, raw_pages, args.query)

    print("\nAnswer:", answer)
    
    print("\nSource Documents:", len(source_docs))
    # Sort source docs by page number for clearer output
    sorted_docs = sorted(source_docs, key=lambda d: d.metadata.get("page", 0) if hasattr(d, "metadata") else 0)
    for i, doc in enumerate(sorted_docs):
        if hasattr(doc, "metadata"):
            print(f"Source {i+1}: {doc.metadata.get('source', 'unknown')} - Page {doc.metadata.get('page', 'unknown')}")
        else:
            print(f"Source {i+1}: Page {i}")

if __name__ == "__main__":
    main()
    