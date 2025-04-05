# emotional-support-chatbot
pre-requirements for vscode to run the chatbot

1-Make sure you're using Python 3.9+
command:python --version
2-Required Python Packages:
command:pip install langchain langchain-community langchain-chroma langchain-huggingface \
    sentence-transformers chromadb pypdf gradio requests pillow
3-For chatgroq
command:pip install langchain-groq
4-If needed (depending on OS or environment):
command:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
5-python-dotenv to load from .env
command:pip install python-dotenv
6-.env file
command:GROQ_API_KEY=gsk_your_key_here
7-Load it in code:
command:from dotenv import load_dotenv
load_dotenv()
8-Enable Virtual Environment (Recommended)
command:python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Linux/Mac
9-VS Code Extensions 
Extensions:Python
Pylance
Jupyter (for notebook-based testing)
Gradio Tools (optional)
10-To Run the App
command:python main.py
