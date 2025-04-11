from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load API keys from .env file
load_dotenv()

# Page setup
st.set_page_config(page_title="AI Research Summarizer", layout="wide")
st.header('📄 AI-Powered Research Summarizer')

# Initialize model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')

# UI Dropdowns
paper_input = st.selectbox("📘 Select type of paper:", 
                           ["Research Article", "Review Paper", "Technical Report", "Thesis", "Whitepaper", "Conference Paper"])

style_input = st.selectbox("📝 Select summary style:", 
                           ["Beginner Friendly", "Technical", "Code Oriented", "Mathematical", "Academic"])

length_input = st.selectbox("📏 Select summary length:", [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (Detailed explanation with subpoints)",
    "Extended (Section-wise breakdown with examples and analogies)",
    "Comprehensive (In-depth explanation with mathematics, context, and practical implications)"
])


user_input = st.text_area("📚 Paste the abstract or full content of the research paper here:")

# Define the Prompt Template
prompt_template = PromptTemplate(
    input_variables=["paper_type", "summary_style", "summary_length", "paper_content"],
    template="""
You are a world-class academic summarization assistant with expertise in scientific, technical, and mathematical domains. 
Your role is to create detailed, accurate, and well-structured summaries of complex research content tailored to various audiences and comprehension levels.

**TASK OVERVIEW:**
Summarize the following *{paper_type}* in a **{summary_style}** style and at a **{summary_length}** detail level. 

**YOUR SUMMARY SHOULD:**
- Be highly accurate and faithful to the original research paper content.
- Clearly outline the objective of the research, methodology, key findings, and their implications.
- Use appropriate terminology, formatting, and domain-specific expressions based on the selected summary style.
- If the selected style is *Mathematical*, provide equations, proofs, or quantitative results where applicable.
- If the style is *Code Oriented*, include relevant pseudocode, programming logic, or computational flow.
- If *Beginner Friendly*, break down technical terms, use analogies, and simplify concepts without losing core meaning.
- For *Academic* or *Technical* styles, maintain a formal tone and include references to methodologies, prior research, and detailed analysis.
- Support the explanation with real-world analogies or comparisons to make complex topics more intuitive.
- Include visual or structural segmentation cues if applicable (e.g., bullet points, subheadings, key takeaway boxes).

**FOR EXTENDED OR COMPREHENSIVE SUMMARIES:**
- Include section-wise breakdowns (e.g., Abstract, Introduction, Methodology, Results, Discussion, Conclusion).
- Discuss the significance of the study in the context of its domain or industry.
- Highlight theoretical foundations, mathematical models, or simulations used.
- Include any limitations, future scope, and potential real-world applications.
- Maintain academic integrity, neutrality, and objective representation of the content.

---

**Research Paper Content:**

\"\"\"{paper_content}\"\"\"

---

Now generate the summary based on the above instructions and selected configurations:
- Paper Type: {paper_type}
- Style: {summary_style}
- Summary Length: {summary_length}
"""
)

# Generate and run the prompt
if st.button('🔍 Generate Summary'):
    if user_input.strip() == "":
        st.warning("⚠️ Please provide the research paper content before summarizing.")
    else:
        prompt = prompt_template.format(
            paper_type=paper_input,
            summary_style=style_input,
            summary_length=length_input,
            paper_content=user_input
        )

        result = model.invoke(prompt)

        st.subheader("📑 Summary Output")
        st.write(result.content)
