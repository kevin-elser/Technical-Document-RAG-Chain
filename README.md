# PDF RAG System

A simple Retrieval-Augmented Generation (RAG) system for querying PDF documents using LangChain and OpenRouter.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   - Edit the `.env` file with your API keys
   - Replace `your_openrouter_key_here` with your actual OpenRouter API key

## Usage

Run the script with a PDF file and your query:

```
python main.py path/to/your/document.pdf "Your question about the document?"
```

### Optional Arguments

- `--model`: Specify the LLM model (default: gpt-3.5-turbo-1106)
- `--temp`: Set the temperature for generation (default: 0.2)

Example with optional arguments:
```
python main.py document.pdf "What is the main topic?" --model "anthropic/claude-2" --temp 0.3
```

## How It Works

1. The PDF is loaded and split into chunks
2. Chunks are embedded using OpenAI embeddings
3. A vector database is created with Chroma
4. When a query is made, the most relevant chunks are retrieved
5. The LLM generates an answer based on the retrieved context

## Customization

You can customize additional parameters in the `.env` file by uncommenting and modifying the optional configuration variables. 