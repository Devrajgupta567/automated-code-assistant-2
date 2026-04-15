# Automated Code Documentation and Debugging Assistant

A Generative AI web app that explains code, detects bugs, generates corrected code, creates API documentation, suggests optimized solutions, and allows users to download the generated analysis as a PDF.

## Team

**Team Name:** Cognitive Coders

**Team Members:** Devraj, Dev, Anubhav

## Domain

Artificial Intelligence

## Features

- Code summarization
- Bug and error detection
- Corrected code generation
- Brief or detailed API documentation
- Optimization suggestions
- Multi-language support
- LangGraph multi-agent workflow
- PDF download of generated analysis

## Tech Stack

- Python
- Streamlit
- Hugging Face Inference API
- LangGraph
- LangChain
- GitHub
- Streamlit Community Cloud

## Required Secret

Add this secret in Streamlit Cloud:

```toml
HF_TOKEN = "your_hugging_face_token_here"
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy this repository on Streamlit Community Cloud and set `HF_TOKEN` in app secrets.
