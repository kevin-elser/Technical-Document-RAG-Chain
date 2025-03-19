# PDF RAG System

A simple Retrieval-Augmented Generation (RAG) system for querying PDF documents using LangChain and OpenRouter. Features both single query usage and chat persistence. Entirely possible to UI-ify this to display the chats and the entire conversation in a single UI instance.

## Example

You can see an example output in the provided `example_output.md` file of the sort of response that can be generated off a query. This response had the embedding model break down the entire 14 page PDF, and the large model was able to search through this data to answer the query (provided as well at the top of the example).

## Setup

1. Install dependencies:
`pip install -r requirements.txt`

2. Configure environment variables:
   - Edit the `.env` file with your API keys
   - Replace `your_openrouter_key_here` with your actual OpenRouter API key

## Usage

### Single Query:

Run the script with a PDF file and your query:
`python main.py path/to/your/document.pdf "Your question about the document?"`

### Conversational Queries:

Start a session:
`python main.py document.pdf "What is this document about?" --session my_session`

Follow up within same session:
`python main.py --session my_session "What are the main chapters?"`

List all sessions:
`python main.py --list-sessions`

### Optional Arguments

- `--model`: Specify the LLM model (default: gpt-4o-mini)
- `--temp`: Set the temperature for generation (default: 0.2)
- `--output`: Output file for the markdown the AI responds with
-`--session`: Name of session to continue

Example with optional arguments:
`python main.py document.pdf "What is the main topic?" --model "anthropic/claude-2" --temp 0.3`

## Customization

You can customize additional parameters in the `.env` file by uncommenting and modifying the optional configuration variables. 