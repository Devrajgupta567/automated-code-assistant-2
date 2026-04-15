from datetime import datetime
import textwrap
from typing import TypedDict

import requests
import streamlit as st
from fpdf import FPDF
from langgraph.graph import END, START, StateGraph


st.set_page_config(
    page_title="Automated Code Documentation and Debugging Assistant",
    layout="wide",
)

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct:fastest"


def get_hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return ""


def call_huggingface_model(prompt, max_tokens=500):
    hf_token = get_hf_token()
    if not hf_token:
        return "Error: HF_TOKEN is missing. Add it in Streamlit Cloud secrets."

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert AI code documentation and debugging assistant. "
                    "Give clear, correct, and structured answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"Error while calling Hugging Face API: {exc}"


def clean_pdf_text(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u2192": "->",
        "\u2713": "OK",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def create_pdf_report(content: str, language: str, doc_style: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 9, "Automated Code Documentation and Debugging Assistant")

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Language: {clean_pdf_text(language)}", ln=True)
    pdf.cell(0, 7, f"Documentation Style: {clean_pdf_text(doc_style)}", ln=True)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "", 11)
    cleaned = clean_pdf_text(content)
    for line in cleaned.splitlines():
        if line.startswith("# "):
            pdf.ln(2)
            pdf.set_font("Arial", "B", 14)
            write_wrapped_pdf_line(pdf, line.replace("# ", ""), width=80)
            pdf.set_font("Arial", "", 11)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Arial", "B", 12)
            write_wrapped_pdf_line(pdf, line.replace("## ", ""), width=85)
            pdf.set_font("Arial", "", 11)
        elif line.strip() == "---":
            pdf.ln(2)
            pdf.cell(0, 1, "", border="T", ln=True)
            pdf.ln(2)
        else:
            write_wrapped_pdf_line(pdf, line, width=95)

    data = pdf.output(dest="S")
    if isinstance(data, str):
        return data.encode("latin-1")
    return bytes(data)


def write_wrapped_pdf_line(pdf: FPDF, line: str, width: int = 70):
    if not line:
        pdf.ln(4)
        return

    printable_width = pdf.w - pdf.l_margin - pdf.r_margin
    chunks = textwrap.wrap(
        line,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    )
    for chunk in chunks or [""]:
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(printable_width, 6, chunk)
        except Exception:
            for start in range(0, len(chunk), 45):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(printable_width, 6, chunk[start : start + 45])


class CodeAssistantState(TypedDict):
    code: str
    language: str
    doc_style: str
    summary: str
    bugs: str
    corrected_code: str
    documentation: str
    optimization: str
    final_answer: str


def summarization_agent(state: CodeAssistantState):
    prompt = f"""
You are a code summarization agent.

Explain the following {state['language']} code.

Code:
{state['code']}

Return:
1. Purpose of the code
2. Main logic
3. Important functions/classes
4. Input and output
"""
    return {"summary": call_huggingface_model(prompt, max_tokens=250)}


def debugging_agent(state: CodeAssistantState):
    prompt = f"""
You are a debugging agent.

Analyze the following {state['language']} code and find bugs, errors, bad practices, or possible runtime issues.

Code:
{state['code']}

Return:
1. Bugs or errors found
2. Why each issue is a problem
3. Corrected version of the code
4. If no bug is found, say "No major bug found"
"""
    return {"bugs": call_huggingface_model(prompt, max_tokens=350)}


def correction_agent(state: CodeAssistantState):
    prompt = f"""
You are a code correction agent.

Fix the following {state['language']} code.

Original Code:
{state['code']}

Debugging Notes:
{state['bugs']}

Return:
1. Corrected code
2. Short explanation of what was fixed
"""
    return {"corrected_code": call_huggingface_model(prompt, max_tokens=350)}


def documentation_agent(state: CodeAssistantState):
    prompt = f"""
You are a documentation generation agent.

Generate {state['doc_style'].lower()} API documentation for this {state['language']} code.

Code:
{state['code']}

If documentation style is Brief:
- Keep it short and beginner-friendly.
- Include only overview, important functions/classes, parameters, and return values.

If documentation style is Detailed:
- Include complete explanation, functions/classes, parameters, return values, examples, notes, edge cases, and limitations.

Return documentation in the selected style.
"""
    return {"documentation": call_huggingface_model(prompt, max_tokens=350)}


def optimization_agent(state: CodeAssistantState):
    prompt = f"""
You are a code optimization agent.

Analyze and optimize this {state['language']} code.

Code:
{state['code']}

Return:
1. Performance improvements
2. Readability improvements
3. Security improvements if needed
4. Optimized code
5. Explanation of optimization
"""
    return {"optimization": call_huggingface_model(prompt, max_tokens=350)}


def final_agent(state: CodeAssistantState):
    return {
        "final_answer": f"""
# Automated Code Documentation and Debugging Assistant Result

## 1. Code Summary

{state['summary']}

---

## 2. Bug/Error Detection

{state['bugs']}

---

## 3. Corrected Code

{state['corrected_code']}

---

## 4. API Documentation ({state['doc_style']})

{state['documentation']}

---

## 5. Optimized Solution

{state['optimization']}
"""
    }


@st.cache_resource
def build_graph():
    graph_builder = StateGraph(CodeAssistantState)
    graph_builder.add_node("summarization_agent", summarization_agent)
    graph_builder.add_node("debugging_agent", debugging_agent)
    graph_builder.add_node("correction_agent", correction_agent)
    graph_builder.add_node("documentation_agent", documentation_agent)
    graph_builder.add_node("optimization_agent", optimization_agent)
    graph_builder.add_node("final_agent", final_agent)

    graph_builder.add_edge(START, "summarization_agent")
    graph_builder.add_edge("summarization_agent", "debugging_agent")
    graph_builder.add_edge("debugging_agent", "correction_agent")
    graph_builder.add_edge("correction_agent", "documentation_agent")
    graph_builder.add_edge("documentation_agent", "optimization_agent")
    graph_builder.add_edge("optimization_agent", "final_agent")
    graph_builder.add_edge("final_agent", END)

    return graph_builder.compile()


code_assistant_graph = build_graph()


def run_langgraph_assistant(code, language, doc_style):
    result = code_assistant_graph.invoke(
        {
            "code": code,
            "language": language,
            "doc_style": doc_style,
            "summary": "",
            "bugs": "",
            "corrected_code": "",
            "documentation": "",
            "optimization": "",
            "final_answer": "",
        }
    )
    return result["final_answer"]


def run_fast_assistant(code, language, doc_style):
    prompt = f"""
Analyze the following {language} code.

Code:
{code}

Return the answer in this exact format:

# 1. Code Summary
Explain what the code does.

# 2. Bug/Error Detection
List bugs, runtime errors, logical errors, security issues, and bad practices.

# 3. Corrected Code
Provide corrected code.

# 4. API Documentation ({doc_style})
Generate {doc_style.lower()} API documentation.

If documentation style is Brief:
- Keep documentation short and include only overview, functions/classes, parameters, and return values.

If documentation style is Detailed:
- Include overview, functions/classes, parameters, return values, example usage, notes, edge cases, and limitations.

# 5. Optimized Solution
Suggest optimized code and explain performance/readability improvements.

Keep the answer clear and concise.
"""
    return call_huggingface_model(prompt, max_tokens=700)


st.title("Automated Code Documentation and Debugging Assistant")
st.write(
    "Paste code and the AI model will summarize it, detect bugs, generate corrected code, "
    "create documentation, and suggest optimized solutions."
)

mode = st.radio(
    "Select Mode",
    ["Fast Mode - Single AI Call", "LangGraph Multi-Agent Mode"],
    index=0,
)

language = st.selectbox(
    "Select Programming Language",
    ["Python", "JavaScript", "Java", "C", "C++", "C#", "Go", "Rust", "PHP", "Ruby"],
)

doc_style = st.radio(
    "What kind of documentation would you like?",
    ["Brief", "Detailed"],
    index=0,
    help="Brief is selected by default if you are not sure.",
)

code = st.text_area(
    "Paste Your Code Here",
    height=320,
    placeholder="Example:\ndef divide(a, b):\n    return a / b",
)

if st.button("Analyze Code"):
    if code.strip() == "":
        st.warning("Please paste code first.")
    else:
        with st.spinner("Analyzing code. Please wait..."):
            if mode == "Fast Mode - Single AI Call":
                result = run_fast_assistant(code, language, doc_style)
            else:
                result = run_langgraph_assistant(code, language, doc_style)

        st.session_state["last_result"] = result
        st.session_state["last_language"] = language
        st.session_state["last_doc_style"] = doc_style
        st.markdown(result)

if "last_result" in st.session_state:
    pdf_bytes = create_pdf_report(
        st.session_state["last_result"],
        st.session_state.get("last_language", language),
        st.session_state.get("last_doc_style", doc_style),
    )
    st.download_button(
        label="Download Analysis as PDF",
        data=pdf_bytes,
        file_name="code_assistant_analysis.pdf",
        mime="application/pdf",
    )
