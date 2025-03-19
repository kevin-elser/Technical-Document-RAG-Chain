from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional
from dotenv import load_dotenv
import datetime
import sqlite3

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
        print(f"\nAnalyzing first {len(first_pages)} pages for TOC:")
        for i, page in enumerate(first_pages):
            print(f"\nPage {i} content preview: {page.page_content[:200]}...")
        
        # Focus on pages that are likely to contain the TOC (usually pages 1-3)
        toc_pages = self.raw_pages[1:4]  # Skip cover page, take next 3 pages
        toc_content = "\n\n".join([f"Page {i+1}: {page.page_content}" for i, page in enumerate(toc_pages)])
        
        # Create a direct prompt for TOC extraction that focuses on numbered sections
        toc_prompt = f"""You are parsing a technical document's table of contents. The document uses a clear numbered structure (e.g., "1. Basic gameplay", "1.1 Basic gameplay", etc.).

Your task is to extract the complete table of contents, paying special attention to:
1. The numbered chapter and section structure (e.g., "1.", "1.1", "1.1.1")
2. The exact chapter/section titles as they appear
3. The page numbers listed for each entry

Format your response as JSON with this structure:
{{
    "toc": [
        {{
            "level": 1,
            "number": "1",
            "title": "Basic gameplay",
            "page": page_number,
            "sections": [
                {{
                    "level": 2,
                    "number": "1.1",
                    "title": "Basic gameplay",
                    "page": page_number,
                    "sections": []
                }}
            ]
        }}
    ]
}}

Important guidelines:
1. Preserve the exact numbering scheme from the document (e.g., "1.", "1.1", "8.3.2")
2. Include ALL sections and subsections
3. Keep the exact section titles as they appear
4. Include page numbers when available
5. Maintain the hierarchical structure based on the numbering
6. Only output valid JSON without any additional text

Document content to parse:
{toc_content}
"""
        
        print("\nExtracting TOC from document...")
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
            print("\nExtracted TOC structure:")
            print(json.dumps(toc_data, indent=2))
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

# Define the state schema for our LangGraph
class RAGState(TypedDict):
    """State for the RAG system"""
    messages: List[Any]  # Chat history containing HumanMessage and AIMessage objects
    documents: Optional[List[Document]]  # Processed documents
    raw_pages: Optional[List[Document]]  # Original document pages
    toc_data: Optional[Dict[str, Any]]  # Table of contents data
    retriever: Optional[Any]  # FAISS retriever
    current_query: Optional[str]  # Current query
    current_response: Optional[str]  # Current response
    source_documents: Optional[List[Document]]  # Source documents for response
    output_file: Optional[str]  # Output file path

def is_toc_query(query):
    """Check if this is a TOC-related query"""
    toc_terms = ["table of contents", "toc", "chapters", "outline", "sections", "what is in this document"]
    return any(term in query.lower() for term in toc_terms)

def setup_rag_retriever(documents, model_name=None):
    """Set up the RAG retriever with document embeddings"""
    # Use HuggingFace embeddings 
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"Using embedding model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create vector database with Document objects
    db = FAISS.from_documents(documents, embeddings)
    
    # Return the retriever with proper configuration
    return db.as_retriever(search_kwargs={"k": 15})

def get_prompt_template():
    """Get the prompt template for the RAG system"""
    return PromptTemplate(
        template="""You are an expert technical document analyst with the ability to understand complex technical information. 
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
9. Use conversation history for context when answering follow-up questions

Previous messages:
{chat_history}

Given the following sections from a technical document:

{context}

Question: {question}

Comprehensive Answer (provide thorough detail and don't omit information):""", 
        input_variables=["context", "question", "chat_history"]
    )

def create_llm(model_name=None, temperature=None, max_tokens=None):
    """Create an LLM instance with the specified parameters"""
    # Get default parameters from environment variables if not provided
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    if temperature is None:
        temperature = float(os.getenv("TEMPERATURE", 0.2))
        
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKENS", 4096))
    
    # Use ChatOpenAI with appropriate configuration
    return ChatOpenAI(
        model_name=model_name, 
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

def get_formatted_messages(messages, max_messages=10):
    """Format messages for prompt context"""
    if not messages:
        return ""
        
    # Take the last few messages
    recent_messages = messages[-max_messages:]
    
    # Format the messages
    formatted = ""
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            formatted += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted += f"Assistant: {message.content}\n\n"
    
    return formatted

def process_query(state: RAGState):
    """Process a query using the RAG system"""
    query = state["current_query"]
    messages = state.get("messages", [])
    
    if not query:
        return {**state, "current_response": "No query provided"}
    
    print(f"Processing query: {query}")
    
    # Handle "please continue" or similar continuation queries
    if query.lower().strip() in ["please continue", "continue", "what's next", "what is next", "go on"]:
        # Look at previous messages to find what chapter we were discussing
        prev_chapter = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                chapter_match = re.search(r'chapter (\d+)', msg.content.lower())
                if chapter_match:
                    prev_chapter = int(chapter_match.group(1))
                    # Look for the next chapter
                    query = f"What is chapter {prev_chapter + 1} about?"
                    break
        
        if not prev_chapter:
            return {**state, "current_response": "I'm not sure what you'd like me to continue with. Could you please specify?"}
    
    # Ensure we have a retriever
    if not state.get("retriever") and state.get("documents"):
        print("Setting up retriever...")
        retriever = setup_rag_retriever(state["documents"])
        state = {**state, "retriever": retriever}
    
    # Check if query is about specific chapters
    chapter_match = re.search(r'chapter (\d+)', query.lower())
    if chapter_match and state.get("toc_data"):
        chapter_num = chapter_match.group(1)
        print(f"Looking for information about Chapter {chapter_num}")
        
        # Find the chapter in TOC
        chapter_info = None
        if isinstance(state["toc_data"], dict) and "toc" in state["toc_data"]:
            for entry in state["toc_data"]["toc"]:
                if entry.get("number") == chapter_num:
                    chapter_info = entry
                    break
        
        if chapter_info:
            # Get the page number and content for this chapter
            page_num = chapter_info.get("page")
            next_chapter_page = None
            
            # Find the next chapter's starting page to know where this chapter ends
            if isinstance(state["toc_data"], dict) and "toc" in state["toc_data"]:
                for entry in state["toc_data"]["toc"]:
                    if entry.get("number") == str(int(chapter_num) + 1):
                        next_chapter_page = entry.get("page")
                        break
            
            if page_num and state.get("raw_pages"):
                # Collect all content from this chapter's pages
                chapter_content = []
                for doc in state["raw_pages"]:
                    doc_page = doc.metadata.get("page")
                    if doc_page >= page_num and (next_chapter_page is None or doc_page < next_chapter_page):
                        chapter_content.append(doc.page_content)
                
                if chapter_content:
                    # Create a response about the chapter
                    chapter_title = chapter_info.get("title", "")
                    sections = chapter_info.get("sections", [])
                    
                    context = f"Chapter {chapter_num}: {chapter_title}\n\n{''.join(chapter_content)}\n\n"
                    if sections:
                        context += "\nSections in this chapter:\n"
                        for section in sections:
                            context += f"- {section.get('number')}: {section.get('title')} (page {section.get('page')})\n"
                    
                    # Get response from LLM about the chapter
                    llm = create_llm()
                    prompt = get_prompt_template().format(
                        context=context,
                        question=f"What is Chapter {chapter_num} about? Please provide a detailed summary.",
                        chat_history=get_formatted_messages(state.get("messages", []))
                    )
                    
                    response = llm.invoke(prompt)
                    result = response.content
                    
                    # Update state with response
                    new_messages = state.get("messages", []) + [
                        HumanMessage(content=query),
                        AIMessage(content=result)
                    ]
                    
                    # Collect all the pages that were part of this chapter
                    chapter_docs = [
                        doc for doc in state["raw_pages"] 
                        if doc.metadata.get("page") >= page_num 
                        and (next_chapter_page is None or doc.metadata.get("page") < next_chapter_page)
                    ]
                    
                    return {
                        **state,
                        "current_response": result,
                        "source_documents": chapter_docs,
                        "messages": new_messages
                    }
    
    # If not a chapter query or chapter not found, proceed with normal RAG
    if not state.get("retriever"):
        return {**state, "current_response": "Retriever not initialized"}
    
    # Get relevant documents
    retriever = state["retriever"]
    relevant_docs = retriever.invoke(query)
    
    # Format context from relevant documents
    context = "\n\n".join([f"[Page {doc.metadata.get('page', 'unknown')}] {doc.page_content}" for doc in relevant_docs])
    
    # Format chat history
    chat_history = get_formatted_messages(state.get("messages", []))
    
    # Get response from LLM
    llm = create_llm()
    prompt = get_prompt_template().format(
        context=context,
        question=query,
        chat_history=chat_history
    )
    
    response = llm.invoke(prompt)
    result = response.content
    
    # Write result to file if specified
    if state.get("output_file"):
        with open(state["output_file"], 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Response written to {state['output_file']}")
    
    # Update state with response
    new_messages = state.get("messages", []) + [
        HumanMessage(content=query),
        AIMessage(content=result)
    ]
    
    return {
        **state,
        "current_response": result,
        "source_documents": relevant_docs,
        "messages": new_messages
    }

def create_rag_graph():
    """Create a LangGraph for the RAG system"""
    workflow = StateGraph(RAGState)
    
    # Add a node for processing queries
    workflow.add_node("process_query", process_query)
    
    # Set the entry point
    workflow.set_entry_point("process_query")
    
    # Add an edge from the node to the end state
    workflow.add_edge("process_query", END)
    
    # Compile the graph
    return workflow.compile()

def load_or_create_document(pdf_path):
    """Load or create a document processor"""
    processor = DocumentProcessor()
    documents, raw_pages, toc_data = processor.load_and_process_documents(pdf_path)
    retriever = setup_rag_retriever(documents)
    
    return documents, raw_pages, toc_data, retriever

def get_session_db_path(session_id):
    """Get the path to the session database"""
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir / f"{session_id}.db"

def list_sessions():
    """List all available sessions"""
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)
    
    sessions = []
    for db_file in sessions_dir.glob("*.db"):
        sessions.append(db_file.stem)
    
    return sessions

def prepare_state_for_serialization(state):
    """Prepare state for serialization by removing non-serializable objects"""
    # Create a shallow copy to avoid modifying the original
    serializable_state = dict(state)
    
    # Replace the retriever with its configuration
    if "retriever" in serializable_state:
        # We don't need to serialize the full retriever - just remove it
        # and we'll recreate it when loaded based on the documents
        serializable_state.pop("retriever", None)
    
    # Keep only serializable parts of documents if present
    if "documents" in serializable_state:
        # Documents should already be serializable, but let's ensure that
        # we aren't storing any complex objects within them
        pass
    
    # Simplify messages to just their content and type
    if "messages" in serializable_state and serializable_state["messages"]:
        simplified_messages = []
        for msg in serializable_state["messages"]:
            if isinstance(msg, HumanMessage):
                simplified_messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                simplified_messages.append({"type": "ai", "content": msg.content})
        serializable_state["_simplified_messages"] = simplified_messages
        serializable_state.pop("messages", None)
    
    return serializable_state

def restore_state_from_serialized(serialized_state):
    """Restore a complete state from a serialized state"""
    # Create a copy to avoid modifying the original
    restored_state = dict(serialized_state)
    
    # Restore messages if simplified versions exist
    if "_simplified_messages" in restored_state:
        messages = []
        for msg in restored_state["_simplified_messages"]:
            if msg["type"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        restored_state["messages"] = messages
        restored_state.pop("_simplified_messages", None)
    
    # Recreate the retriever if we have documents
    if "documents" in restored_state and not restored_state.get("retriever"):
        if restored_state["documents"]:
            # This will recreate the retriever from documents
            retriever = setup_rag_retriever(restored_state["documents"])
            restored_state["retriever"] = retriever
    
    return restored_state

def run_query(pdf_path, query, session_id=None, output_file=None, model=None, temperature=None, max_tokens=None):
    """Run a query against a document, using an optional session for context"""
    # Create the graph
    graph = create_rag_graph()
    
    # Create the state saver if using a session
    if session_id:
        # Get the database path
        db_path = get_session_db_path(session_id)
        db_path_str = str(db_path)
        
        try:
            # Create a connection and initialize the database
            conn = sqlite3.connect(db_path_str)
            saver = SqliteSaver(conn)
            
            # Initialize the database schema
            saver.setup()
            
            # Set up config for session
            config = {"configurable": {"thread_id": session_id, "checkpoint_ns": session_id}}
            print(f"Config being used for get_tuple: {config}") # DEBUG PRINT
            try:
                print(f"Attempting to load session: {session_id}")
                # For debugging purposes, let's check what tables exist
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f"Tables in database: {tables}")
                
                # Retrieve the previous state using get_tuple
                try:
                    checkpoint_tuple = saver.get_tuple(config)
                    print(f"Checkpoint tuple retrieved: {checkpoint_tuple is not None}")
                    print(f"Checkpoint data: {checkpoint_tuple.checkpoint if checkpoint_tuple else None}")
                    
                    # Check if we have a valid checkpoint with the correct structure
                    if (checkpoint_tuple and 
                        hasattr(checkpoint_tuple, 'checkpoint') and 
                        isinstance(checkpoint_tuple.checkpoint, dict)):
                        
                        # Get the serialized state from channel_values
                        serialized_state = checkpoint_tuple.checkpoint.get("channel_values", {}).get("default", {})
                        if serialized_state:
                            previous_state = restore_state_from_serialized(serialized_state)
                            print(f"Loaded session state: {session_id}")
                            
                            # If we have a new PDF path, reload the document
                            if pdf_path and (not previous_state or 
                                            previous_state.get("_metadata", {}).get("pdf_path") != pdf_path):
                                print(f"Loading new document from: {pdf_path}")
                                documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
                                
                                # Initialize state with new document
                                state = {
                                    "messages": previous_state.get("messages", []) if previous_state else [],
                                    "documents": documents,
                                    "raw_pages": raw_pages,
                                    "toc_data": toc_data,
                                    "retriever": retriever,
                                    "current_query": query,
                                    "output_file": output_file,
                                    "_metadata": {"pdf_path": pdf_path}
                                }
                            else:
                                # Use existing session with new query
                                state = {
                                    **previous_state,
                                    "current_query": query,
                                    "output_file": output_file
                                }
                        else:
                            print("No valid serialized state found in checkpoint. Creating new state...")
                            documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
                            state = {
                                "messages": [],
                                "documents": documents,
                                "raw_pages": raw_pages,
                                "toc_data": toc_data,
                                "retriever": retriever,
                                "current_query": query,
                                "output_file": output_file,
                                "_metadata": {"pdf_path": pdf_path}
                            }
                    else:
                        print("Invalid checkpoint structure. Creating new state...")
                        documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
                        state = {
                            "messages": [],
                            "documents": documents,
                            "raw_pages": raw_pages,
                            "toc_data": toc_data,
                            "retriever": retriever,
                            "current_query": query,
                            "output_file": output_file,
                            "_metadata": {"pdf_path": pdf_path}
                        }
                except Exception as e:
                    print(f"Error retrieving checkpoint tuple: {str(e)}")
                    documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
                    state = {
                        "messages": [],
                        "documents": documents,
                        "raw_pages": raw_pages,
                        "toc_data": toc_data,
                        "retriever": retriever,
                        "current_query": query,
                        "output_file": output_file,
                        "_metadata": {"pdf_path": pdf_path}
                    }
                
                # Run the graph with the appropriate config
                final_state = graph.invoke(state, config=config)
                
                # Save the state
                try:
                    print(f"Attempting to save session state: {session_id}")
                    print(f"Config being used for put: {config}") # DEBUG PRINT
                    metadata = {
                        "source": "rag_query", 
                        "timestamp": str(datetime.datetime.now()),
                        "query": query
                    }
                    
                    # Prepare the state for serialization
                    serialized_state = prepare_state_for_serialization(final_state)
                    
                    checkpoint = {
                        "v": 1,  # version
                        "id": session_id,  # session ID
                        "ts": str(datetime.datetime.now()),  # timestamp
                        "channel_values": {
                            "default": serialized_state
                        },
                        "channel_versions": {
                            "default": 1
                        },
                        "versions_seen": {},
                        "pending_sends": []
                    }
                    
                    saver.put(
                        config=config,
                        checkpoint=checkpoint,
                        metadata=metadata,
                        new_versions={"default": 1}
                    )
                    print(f"Session state saved: {session_id}")
                    
                except Exception as e:
                    print(f"Error saving session state: {str(e)}")
                    print(f"Type of error: {type(e)}")
                    print(f"Full error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
                
                # Close the connection
                conn.close()
                return final_state.get("current_response", ""), final_state.get("source_documents", [])
                
            except Exception as e:
                print(f"Error reading session: {str(e)}")
                conn.close()
                # Create new state if session read fails
                documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
                state = {
                    "messages": [],
                    "documents": documents,
                    "raw_pages": raw_pages,
                    "toc_data": toc_data,
                    "retriever": retriever,
                    "current_query": query,
                    "output_file": output_file,
                    "_metadata": {"pdf_path": pdf_path}
                }
                config = {}
                
                # Run the graph without session
                final_state = graph.invoke(state, config=config)
                return final_state.get("current_response", ""), final_state.get("source_documents", [])
                
        except Exception as e:
            print(f"Error setting up session: {str(e)}")
            print("Continuing without session persistence...")
            
            # Fall back to non-session approach
            documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
            state = {
                "messages": [],
                "documents": documents,
                "raw_pages": raw_pages,
                "toc_data": toc_data,
                "retriever": retriever,
                "current_query": query,
                "output_file": output_file
            }
            config = {}
            
            # Run the graph without session
            final_state = graph.invoke(state, config=config)
            return final_state.get("current_response", ""), final_state.get("source_documents", [])
    else:
        # No session, just load the document and create a one-time state
        documents, raw_pages, toc_data, retriever = load_or_create_document(pdf_path)
        state = {
            "messages": [],
            "documents": documents,
            "raw_pages": raw_pages,
            "toc_data": toc_data,
            "retriever": retriever,
            "current_query": query,
            "output_file": output_file
        }
        config = {}
        
        # Run the graph without session
        final_state = graph.invoke(state, config=config)
        return final_state.get("current_response", ""), final_state.get("source_documents", [])

def main():
    parser = argparse.ArgumentParser(description="Technical Document RAG System")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF document")
    parser.add_argument("query", nargs="?", help="The question to ask")
    parser.add_argument("--model", help="The LLM model to use (overrides .env setting)")
    parser.add_argument("--temp", type=float, help="The temperature for the LLM (overrides .env setting)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens in response (overrides .env setting)")
    parser.add_argument("--output", help="Output file for the response (in markdown format)")
    parser.add_argument("--session", help="Session ID to maintain conversation context")
    parser.add_argument("--list-sessions", action="store_true", help="List all available sessions")
    args = parser.parse_args()
    
    # List sessions if requested
    if args.list_sessions:
        sessions = list_sessions()
        if sessions:
            print("Available sessions:")
            for session in sessions:
                print(f"  - {session}")
        else:
            print("No sessions found.")
        return
    
    # Validate parameters
    if not args.session and (not args.pdf_path or not args.query):
        parser.print_help()
        return
    
    if args.session and not args.query:
        print("Error: No query provided for session.")
        return
    
    if args.session and not args.pdf_path:
        # Check if session exists before proceeding
        db_path = get_session_db_path(args.session)
        # Use os.path.exists instead of Path.exists() for better compatibility
        if not os.path.exists(str(db_path)):
            print(f"Error: Session '{args.session}' does not exist and no PDF path provided.")
            return
    
    # Process the query
    print(f"Processing query with {'' if not args.session else f'session {args.session}: '}{args.query}")
    
    answer, source_docs = run_query(
        args.pdf_path, 
        args.query, 
        session_id=args.session,
        output_file=args.output,
        model=args.model,
        temperature=args.temp,
        max_tokens=args.max_tokens
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
    