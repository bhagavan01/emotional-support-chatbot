from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr
import json
from datetime import datetime
import requests
from PIL import Image
import io

# Global variable to store conversation history
conversation_history = []

# Download LPU logo
def get_lpu_logo():
    logo_url = "https://www.lpu.in/images/logo.png"
    response = requests.get(logo_url)
    return Image.open(io.BytesIO(response.content))

def save_conversation_history():
    with open("conversation_history.json", "w") as f:
        json.dump(conversation_history, f)

def load_conversation_history():
    global conversation_history
    try:
        with open("conversation_history.json", "r") as f:
            conversation_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conversation_history = []

def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY", "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")
    return ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )

def create_vector_db():
    pdf_path = r"C:\Users\Bhagavan\Downloads\emotional support.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """You are a compassionate mental health counselor. Provide thoughtful, empathetic responses to mental health concerns.
    Use the following context to help answer the question:
    {context}
    
    Current conversation:
    User: {question}
    Counselor: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def update_history(user_input, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_history.append({
        "timestamp": timestamp,
        "user": user_input,
        "bot": bot_response
    })
    save_conversation_history()
    return conversation_history

print("Initializing Chatbot.........")
try:
    # Load LPU logo
    lpu_logo = get_lpu_logo()
    lpu_logo_path = "lpu_logo.png"
    lpu_logo.save(lpu_logo_path)
    
    load_conversation_history()
    llm = initialize_llm()
    
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(db_path):
        print("Creating new vector database...")
        vector_db = create_vector_db()
    else:
        print("Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    
    qa_chain = setup_qa_chain(vector_db, llm)
    
    def chatbot_response(message, chat_history):
        if not message.strip():
            return "Please share what's on your mind. I'm here to listen."
        
        try:
            result = qa_chain({"query": message})
            response = f"I understand. {result['result']}\n\nRemember you're not alone in this."
            update_history(message, response)
            return response
        except Exception as e:
            error_msg = f"I'm having trouble responding. Please try again later. ({str(e)})"
            update_history(message, error_msg)
            return error_msg

    # Dark Theme CSS with expanded history
    custom_css = """
    .gradio-container {
        background: #121212 !important;
        color: #ffffff !important;
    }
    .chatbot {
        min-height: 650px;
        border-radius: 12px !important;
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
    }
    /* History section styling */
    .history-column {
        min-height: 650px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .history-accordion {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .history-accordion > .wrap {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .history-panel {
        flex-grow: 1 !important;
        overflow-y: auto !important;
        max-height: 550px !important;
    }
    .chatbot .user {
        background: #2d3748 !important;
        color: white !important;
        border-radius: 12px 12px 0 12px !important;
        max-width: 85%;
        margin-left: auto;
        border: 1px solid #4a5568 !important;
    }
    .chatbot .assistant {
        background: #2d3748 !important;
        color: white !important;
        border-radius: 12px 12px 12px 0 !important;
        max-width: 85%;
        border: 1px solid #4a5568 !important;
    }
    .team-details {
        background: #1e1e1e !important;
        color: white !important;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #333 !important;
        margin-top: 10px;
    }
    .history-panel {
        background: #1e1e1e !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid #333 !important;
    }
    .examples-container {
        gap: 8px !important;
    }
    .example {
        background: #2d3748 !important;
        color: white !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .example:hover {
        background: #4a5568 !important;
        transform: translateY(-2px) !important;
    }
    .header {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .footer {
        background: #1e1e1e !important;
        color: white !important;
        padding: 16px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #333 !important;
    }
    .btn-primary {
        background: #4a5568 !important;
        color: white !important;
        border: none !important;
    }
    .btn-primary:hover {
        background: #2d3748 !important;
    }
    .btn-secondary {
        background: #1e1e1e !important;
        color: white !important;
        border: 1px solid #4a5568 !important;
    }
    .btn-secondary:hover {
        background: #2d3748 !important;
    }
    .textbox {
        border-radius: 12px !important;
        padding: 12px !important;
        background: #1e1e1e !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    label {
        color: white !important;
    }
    .accordion {
        background: #1e1e1e !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Poppins"), "Arial", "sans-serif"]
    ), title="LPU Mental Health Chatbot", css=custom_css) as app:
        
        # Header section
        with gr.Row(equal_height=True, variant="panel"):
            with gr.Column(scale=2):
                gr.Markdown("""
                <div class="header">
                <h1 style="margin: 0;">🧠 LPU Emotional Support Companion</h1>
                <p style="margin: 0; opacity: 0.9;">A safe space to share your thoughts and feelings</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=150):
                gr.Image(lpu_logo_path, label="Lovely Professional University", 
                        width=120, show_label=False, elem_classes="logo")
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("""
                <div class="team-details">
                <h3 style="margin-top: 0;">Team Details</h3>
                <p>A. Bhagavan (12304524)</p>
                <p>Santhan (12304524)</p>
                <p>Shiva (12304524)</p>
                </div>
                """)
        
        # Main content area
        with gr.Row():
            # Left sidebar with history (now expanded)
            with gr.Column(scale=1, min_width=300, elem_classes="history-column"):
                with gr.Accordion("📚 Conversation History", open=True, elem_classes="history-accordion"):
                    history_display = gr.JSON(
                        value=conversation_history,
                        label="Past Conversations",
                        container=True,
                        elem_classes="history-panel"
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        new_chat_btn = gr.Button("🆕 New Chat", variant="primary")
                        
                def clear_history():
                    global conversation_history
                    conversation_history = []
                    save_conversation_history()
                    return []
                    
                clear_btn.click(
                    fn=clear_history,
                    outputs=history_display
                )
                
                def new_chat():
                    return []
                    
                new_chat_btn.click(
                    fn=new_chat,
                    outputs=[history_display]
                )
            
            # Main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=650,  # Increased height to match history
                    elem_classes="chatbot",
                    show_copy_button=True,
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
                        "https://cdn-icons-png.flaticon.com/512/4712/4712139.png"
                    )
                )
                msg = gr.Textbox(
                    label="Type your message here...", 
                    placeholder="Share what's on your mind...",
                    elem_classes="textbox",
                    container=False
                )
                clear = gr.ClearButton([msg, chatbot], variant="secondary")
                
                examples = gr.Examples(
                    examples=[
                        ["I can't sleep at night"],
                        ["I've been feeling really anxious"],
                        ["How do I deal with negative thoughts?"],
                        ["I'm feeling overwhelmed with studies"],
                        ["I don't feel motivated anymore"]
                    ],
                    inputs=msg,
                    label="Try these examples:",
                    examples_per_page=5
                )
                
                def respond(message, chat_history):
                    bot_message = chatbot_response(message, chat_history)
                    chat_history.append((message, bot_message))
                    update_history(message, bot_message)
                    return "", chat_history
                
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                
                # Footer with resources
                gr.Markdown("""
                <div class="footer">
                <h3 style="margin-top: 0;">Crisis Resources</h3>
                <p>• LPU Counseling Center: <a href="#" style="color: #a3bffa;">Contact Info</a></p>
                <p>• National Suicide Prevention Lifeline: 988 (US)</p>
                <p>• Crisis Text Line: Text HOME to 741741</p>
                <p style="font-size: 0.8em; opacity: 0.7; margin-bottom: 0;">This is not a substitute for professional help.</p>
                </div>
                """)
        
        chatbot.change(
            fn=lambda: conversation_history,
            outputs=history_display
        )
    
    app.launch()

except Exception as e:
    print(f"Failed to initialize chatbot: {str(e)}")