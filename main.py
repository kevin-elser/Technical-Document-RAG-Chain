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
import re
import json
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

class DocumentProcessor:
    def __init__(self):
        self.toc_data = None
        self.raw_pages = None
        self.documents = None
        self.embedding_model = None
        self.vector_db = None
    
    def load_and_process_documents(self, pdf_path):
        print(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        self.raw_pages = loader.load()
        print(f"Loaded {len(self.raw_pages)} pages from PDF")
        
        # First, extract TOC from early pages
        self.toc_data = self.extract_toc()
        
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
        self.documents = []
        for doc in self.raw_pages:
            page_num = doc.metadata.get("page", 0)
            chunks = text_splitter.split_text(doc.page_content)
            
            # Add more detailed metadata to each chunk
            for i, chunk in enumerate(chunks):
                # Add TOC metadata if we have it (helps with context)
                toc_context = self.get_toc_context_for_page(page_num) if self.toc_data else ""
                
                metadata = {
                    "source": doc.metadata.get("source"),
                    "page": page_num,
                    "chunk": i,
                    "content_type": "page_content",
                    "toc_context": toc_context
                }
                self.documents.append(Document(page_content=chunk, metadata=metadata))
        
        print(f"Document processed into {len(self.documents)} chunks")
        return self.documents, self.raw_pages, self.toc_data
    
    def extract_toc(self):
        """Extract table of contents from the first several pages"""
        if not self.raw_pages or len(self.raw_pages) < 3:
            return None
            
        # Initialize OpenAI model for TOC extraction
        llm = ChatOpenAI(
            model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo-1106"), 
            temperature=0.0,  # Use 0 temperature for deterministic extraction
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        first_pages = self.raw_pages[:10]
        toc_content = "\n\n".join([f"Page {i}: {page.page_content}" for i, page in enumerate(first_pages)])
        
        # Create a direct prompt for TOC extraction
        toc_prompt = f"""Extract the complete table of contents from this document, including all 
        chapters, sections, and subsections with their page numbers if available.
        
        Format your response as JSON with the following structure:
        {{
            "toc": [
                {{
                    "level": 1,  
                    "title": "Chapter title",
                    "page": page_number,
                    "sections": [
                        {{
                            "level": 2,
                            "title": "Section title",
                            "page": page_number
                            "sections": [ ... ]
                        }}
                    ]
                }}
            ]
        }}
        
        If you can't find a complete TOC, extract as much structured information as possible about the document's organization.
        Only output valid JSON without any additional text.
        
        Document content:
        {toc_content}
        """
        
        print("Extracting TOC from document...")
        try:
            response = llm.invoke(toc_prompt)
            
            # Clean and parse the response
            content = response.content.strip()
            # Remove any markdown code block indicators if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            toc_data = json.loads(content)
            print("Successfully extracted TOC")
            return toc_data
        except Exception as e:
            print(f"Error extracting TOC: {e}")
            return None
    
    def get_toc_context_for_page(self, page_num):
        """Get relevant TOC context for a specific page"""
        if not self.toc_data or not isinstance(self.toc_data, dict) or "toc" not in self.toc_data:
            return ""
        
        def extract_relevant_entries(entries, current_path="", results=None):
            if results is None:
                results = []
            
            for entry in entries:
                # Skip if no page number
                if "page" not in entry or entry["page"] is None:
                    continue
                    
                # Convert page to int if it's a string
                entry_page = entry["page"]
                if isinstance(entry_page, str) and entry_page.isdigit():
                    entry_page = int(entry_page)
                elif not isinstance(entry_page, int):
                    continue
                
                # Check if page is close to our target page
                if abs(entry_page - page_num) <= 5:  # Within 5 pages
                    level = entry.get("level", 1)
                    indent = "  " * (level - 1)
                    title = entry.get("title", "")
                    path = f"{current_path} > {title}" if current_path else title
                    results.append(f"{indent}{title} (p.{entry_page}) [{path}]")
                
                # Check subsections
                if "sections" in entry and isinstance(entry["sections"], list):
                    new_path = f"{current_path} > {entry.get('title', '')}" if current_path else entry.get("title", "")
                    extract_relevant_entries(entry["sections"], new_path, results)
            
            return results
        
        relevant_entries = extract_relevant_entries(self.toc_data["toc"])
        return "\n".join(relevant_entries)

def is_toc_query(query):
    """Check if this is a TOC-related query"""
    toc_terms = ["table of contents", "toc", "chapters", "outline", "sections", "what is in this document"]
    return any(term in query.lower() for term in toc_terms)

def setup_rag_chain(documents, toc_data=None, model_name=None, temperature=None, max_tokens=None):
    # Get default parameters from environment variables if not provided
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    if temperature is None:
        temperature = float(os.getenv("TEMPERATURE", 0.2))
        
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKENS", 4096))
    
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
    6. Always provide COMPLETE AND COMPREHENSIVE information - do not truncate your response
    7. When listing chapters or sections, ensure you include ALL of them that appear in the context
    8. Format code blocks, tables, and technical details properly with markdown

    Given the following sections from a technical document:

    {context}

    Question: {question}

    Comprehensive Answer (provide thorough detail and don't omit information):"""
    
    # Create prompt without TOC info to avoid input validation issues
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Use ChatOpenAI with appropriate configuration
    llm = ChatOpenAI(
        model_name=model_name, 
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    # Configure the chain to use the retriever and prompt
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={"k": 15}),  # Increased from 12 to get more context
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs
    )

    return qa, llm, toc_data

def run_query(qa_chain, llm, raw_pages, toc_data, query, output_file=None):
    print(f"Processing query: {query}")
    
    # Special handling for TOC queries - use extracted TOC if available
    if is_toc_query(query) and toc_data:
        print("Detected TOC query - using extracted TOC data")
        
        # Create a comprehensive TOC description from the extracted data
        toc_description = "# Table of Contents\n\n"
        
        def format_toc_for_display(entries, level=0):
            toc_text = ""
            for entry in entries:
                indent = "  " * level
                title = entry.get("title", "Unknown")
                page = entry.get("page", "")
                page_text = f" (page {page})" if page else ""
                toc_text += f"{indent}- **{title}**{page_text}\n"
                
                if "sections" in entry and isinstance(entry["sections"], list):
                    toc_text += format_toc_for_display(entry["sections"], level + 1)
            return toc_text
        
        if isinstance(toc_data, dict) and "toc" in toc_data:
            toc_description += format_toc_for_display(toc_data["toc"])
        else:
            toc_description += "Could not extract a structured table of contents from this document."
        
        # Add additional information prompt for more comprehensive TOC response
        toc_prompt = f"""
        You are analyzing a technical document and need to present its table of contents in a clear,
        structured format. Here is the extracted table of contents information:
        
        {toc_description}
        
        Please format this information as a complete, well-structured table of contents.
        Add a brief introduction about what this document appears to cover based on the chapter titles.
        Format your response using proper markdown with headings, bullet points, and indentation to show the hierarchy.
        If specific page numbers are available, include them.
        """
        
        response = llm.invoke(toc_prompt)
        result = response.content
        sources = raw_pages[:10]  # First 10 pages as sources
    else:
        # For regular queries, use the QA chain with only the required inputs
        # The TOC data will be incorporated into the context through the document metadata
        
        # Create the standard input dict without toc_info
        input_dict = {"query": query}
        
        # If we have TOC data, enhance the query with relevant information
        enhanced_query = query
        if toc_data and isinstance(toc_data, dict) and "toc" in toc_data:
            # Extract page numbers from the query if present
            page_refs = re.findall(r'pages? (\d+)(?:\s*-\s*(\d+))?', query, re.IGNORECASE)
            
            if page_refs:
                # Add TOC context for those specific pages to the query
                for start_page, end_page in page_refs:
                    start = int(start_page)
                    end = int(end_page) if end_page else start
                    
                    relevant_toc = []
                    
                    # Find TOC entries relevant to these pages
                    def find_relevant_toc_entries(entries):
                        results = []
                        for entry in entries:
                            page = entry.get("page")
                            if page is not None:
                                if isinstance(page, str) and page.isdigit():
                                    page = int(page)
                                if isinstance(page, int) and start <= page <= end:
                                    results.append(f"{entry.get('title', 'Unknown')} (p.{page})")
                            
                            if "sections" in entry and isinstance(entry["sections"], list):
                                results.extend(find_relevant_toc_entries(entry["sections"]))
                        return results
                    
                    if "toc" in toc_data:
                        relevant_toc = find_relevant_toc_entries(toc_data["toc"])
                    
                    # If we found relevant TOC entries, enhance the query
                    if relevant_toc:
                        relevant_context = "\n".join(relevant_toc)
                        print(f"Adding relevant TOC context for pages {start}-{end}")
            
        
        response = qa_chain.invoke(input_dict)
        result = response["result"]
        sources = response["source_documents"]
    
    # Write the result to output file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
            
        print(f"Response written to {output_file}")
    
    return result, sources

def main():
    parser = argparse.ArgumentParser(description="Technical Document RAG System")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--model", help="The LLM model to use (overrides .env setting)")
    parser.add_argument("--temp", type=float, help="The temperature for the LLM (overrides .env setting)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens in response (overrides .env setting)")
    parser.add_argument("--output", help="Output file for the response (in markdown format)")
    args = parser.parse_args()

    processor = DocumentProcessor()
    documents, raw_pages, toc_data = processor.load_and_process_documents(args.pdf_path)
    
    print(f"Setting up RAG chain with model: {args.model or os.getenv('MODEL_NAME', 'gpt-3.5-turbo-1106')}")
    qa_chain, llm, toc_data = setup_rag_chain(
        documents, 
        toc_data=toc_data,
        model_name=args.model, 
        temperature=args.temp,
        max_tokens=args.max_tokens
    )
    
    print(f"Querying: {args.query}")
    answer, source_docs = run_query(
        qa_chain, 
        llm, 
        raw_pages, 
        toc_data,
        args.query,
        output_file=args.output
    )

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
    