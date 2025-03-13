# Document Chat Bot

A Streamlit-based chatbot that processes both PDF and DOCX documents (Russian and English) and answers questions about their content using GPT-4 and FAISS vector database.

## Features
- Multi-document support (PDF and DOCX)
- Automatic Russian to English translation
- Advanced text chunking with overlap
- FAISS vector similarity search
- GPT-4 for accurate responses
- Token usage tracking
- Debug information panel

## Project Structure
```
streamlit_bot/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (API keys)
└── .streamlit/        # Streamlit configuration
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
Create a `.env` file and add your API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your PDF or DOCX documents through the Streamlit interface
2. Click "Process" to analyze the documents
3. Start chatting with your documents using the text input field
4. View token usage and debug information in the expandable sections

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your OpenAI API key to Streamlit secrets
5. Deploy!
