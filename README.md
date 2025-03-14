
# LLM Chatbot and Document Summarization Repository

This repository contains two Python scripts designed to interact with and utilize Large Language Models (LLMs) for different purposes:

- **`llama_summerize.py`**: A Streamlit application for document summarization and chatbot interactions using the `llama2` model via Ollama and HuggingFace models. It allows users to upload documents (PDF, TXT, DOCX) and ask questions about their content or get summaries.
- **`llm_chatbot.py`**: A simple command-line chatbot powered by `LlamaCpp` and a local GGUF model (`Llama-2-7B-Chat-GGUF`). It provides a basic conversational interface directly in your terminal.

## Features

**`llama_summerize.py`:**

- **Document Summarization:**
    - Supports uploading multiple files in PDF, TXT, and DOCX formats.
    - Extracts text from uploaded documents.
    - Generates summaries of the document content using the `llama2` model (via Ollama) or HuggingFace models (commented out in the code).
    - Calculates and displays cosine similarity as a measure of "accuracy" between the original text and the generated summary.
- **Document Chatbot:**
    - Creates a conversational chatbot interface within Streamlit.
    - Processes uploaded documents to create a knowledge base.
    - Allows users to ask questions about the content of the uploaded documents.
    - Maintains chat history for context-aware conversations.
    - Uses `FAISS` for efficient vector storage and retrieval of document chunks.
    - Employs `sentence-transformers/all-MiniLM-L6-v2` for text embeddings.
- **Input Options:**
    - Provides options for both text input and file uploads for summarization.
- **User-Friendly Interface:**
    - Built with Streamlit for an interactive and easy-to-use web interface.

**`llm_chatbot.py`:**

- **Command-Line Chatbot:**
    - Simple and direct chatbot interface in the terminal.
    - Uses `LlamaCpp` to run the `Llama-2-7B-Chat-GGUF` model locally.
    - Takes user input from the command line and generates responses.
    - Demonstrates basic question-answering capabilities of the LLM.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd llm-chatbot
   ```

2. **Install Python dependencies:**
   It is recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows

   pip install -r requirements.txt
   ```

   **Note:** Ensure you have `llama-cpp-python` installed correctly, especially if you intend to use `llm_chatbot.py`. Installation might require specific configurations depending on your system and GPU setup. Refer to the `llama-cpp-python` documentation for detailed instructions.

3. **Install Ollama (if you plan to use `llama_summerize.py` with Ollama):**

   - Follow the installation instructions for your operating system from the [Ollama website](https://ollama.com/).
   - Once installed, ensure Ollama is running and you have pulled the `llama2` model.
     ```bash
     ollama pull llama2
     ```

4. **Download the `Llama-2-7B-Chat-GGUF` model (if you plan to use `llm_chatbot.py`):**

   - You need to download a GGUF format model for `LlamaCpp`. A suitable model like `Llama-2-7B-Chat-GGUF` can be downloaded from Hugging Face or other model repositories.
   - Place the downloaded GGUF model file (e.g., `llama-2-7b-chat.Q4_K_M.gguf`) in the `model/` directory of this repository. You might need to create the `model/` directory if it doesn't exist.

## Usage

### Running `llama_summerize.py` (Streamlit App)

1. **Ensure Ollama is running in the background (if using Ollama).**
2. **Navigate to the repository directory in your terminal.**
3. **Run the Streamlit application:**
   ```bash
   streamlit run llama_summerize.py
   ```
4. **Open the Streamlit app in your browser.** The URL will usually be displayed in your terminal (e.g., `http://localhost:8501`).
5. **Use the application:**
   - **Summarization:**
     - Choose between "Text Input" or "File Upload".
     - If "Text Input", enter your text prompt and click "Generate Response".
     - If "File Upload", upload one or more files (PDF, TXT, DOCX). Summaries will be generated and displayed for each document.
   - **Chatbot:**
     - Upload files to create the chatbot knowledge base.
     - Type your questions in the "Question" text input and click "Send".
     - Chat history will be displayed, showing both your questions and the model's responses.

### Running `llm_chatbot.py` (Command-Line Chatbot)

1. **Ensure you have downloaded the `Llama-2-7B-Chat-GGUF` model and placed it in the `model/` directory.**
2. **Navigate to the repository directory in your terminal.**
3. **Run the script:**
   ```bash
   python llm_chatbot.py
   ```
4. **Start chatting:**
   - You will see the prompt `>` indicating the chatbot is ready for input.
   - Type your question and press Enter.
   - The model's response will be printed in the terminal.
   - Continue asking questions as needed.
   - To exit, you can typically use `Ctrl+C`.

## Scripts Description

- **`llama_summerize.py`**: This script provides a comprehensive interface for interacting with LLMs for document processing. It leverages Streamlit to create a user-friendly web application. It offers two main functionalities: document summarization and a document-based chatbot. For summarization, it can process direct text input or uploaded files, providing summaries and a cosine similarity score as a rough measure of accuracy. The chatbot feature allows users to upload documents which are then processed and used as context for answering questions, providing a conversational experience related to the document content. It primarily uses Ollama for running the `llama2` model but also includes commented-out code for using HuggingFace models, offering flexibility in model selection.

- **`llm_chatbot.py`**: This script is a simpler implementation focused on demonstrating a basic command-line chatbot. It utilizes `LlamaCpp` to directly run the `Llama-2-7B-Chat-GGUF` model locally. This approach is beneficial for users who want to run chatbots offline and have more control over model execution. The script sets up a simple loop to take user questions from the command line and print the model's responses, showcasing the fundamental conversational capabilities of the LLM.

## Dependencies

- **Python Libraries (install using `pip install -r requirements.txt`):**
  - `streamlit`
  - `streamlit-chat`
  - `pdfplumber`
  - `docx`
  - `langchain`
  - `langchain_community`
  - `huggingface_hub`
  - `transformers`
  - `sentence-transformers`
  - `faiss-cpu`
  - `scikit-learn`
  - `llama-cpp-python` (for `llm_chatbot.py`)
  - `python-dotenv` (commented out, but might be used for environment variables)

- **Software:**
  - **Ollama** (if using `llama_summerize.py` with Ollama) - [https://ollama.com/](https://ollama.com/)
  - **Llama-2-7B-Chat-GGUF model file** (if using `llm_chatbot.py`) - Download from a trusted source like Hugging Face.

## Model Requirements

- **`llama_summerize.py`**:
  - By default configured to use `llama2` model via **Ollama**. Ensure Ollama is installed and the `llama2` model is pulled (`ollama pull llama2`).
  - Optionally, can be configured to use HuggingFace models (code for this is commented out and would require a Hugging Face API token and uncommenting/modifying relevant parts of the code).

- **`llm_chatbot.py`**:
  - Requires the **`Llama-2-7B-Chat-GGUF`** model file placed in the `model/` directory. This script is specifically designed to work with this GGUF model through `LlamaCpp`.

## Potential Improvements

- **Error Handling:** Implement more robust error handling, especially for file processing and model loading.
- **Advanced Summarization Techniques:** Explore more sophisticated summarization methods beyond basic chunking.
- **Prompt Engineering:** Refine prompts for both summarization and chatbot functionalities to improve response quality.
- **UI Enhancements (for `llama_summerize.py`):** Improve the Streamlit UI with features like progress indicators, better layout, and more user customization options.
- **Model Configuration:** Allow users to configure model parameters (e.g., temperature, top_p) in both scripts.
- **More Model Options:** Expand model support to include other LLMs in both scripts, providing users with more choices.
- **Evaluation Metrics:** Implement more comprehensive evaluation metrics for summarization and chatbot performance beyond cosine similarity.

This repository provides a starting point for exploring document summarization and chatbot applications using LLMs. You can further develop and customize these scripts to suit your specific needs.
```

