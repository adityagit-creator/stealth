import pandas as pd
import numpy as np
from langchain_openai import OpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
from typing import Dict, List, Any
import warnings
import gradio as gr
from dotenv import load_dotenv

# Ignore warnings for a cleaner interface
warnings.filterwarnings('ignore')
# Load environment variables from .env file
load_dotenv()

class ExcelAIQuerySystem:
    """
    A system to query Excel files using natural language, powered by OpenAI and LangChain.
    """
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = OpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.excel_data = {}
        self.sheet_descriptions = {}
        self.vectorstore = None
        self.logs = []

    def load_excel_file(self, file_path: str) -> str:
        """Loads and processes an Excel file, generating descriptions and a vector store."""
        self.logs.clear()
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            self.logs.append(f"✅ Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")

            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    df = self._clean_dataframe(df)
                    self.excel_data[sheet_name] = df

                    description = self._generate_sheet_description(sheet_name, df)
                    self.sheet_descriptions[sheet_name] = description
                    self.logs.append(f"  - Loaded and described sheet '{sheet_name}' ({df.shape[0]} rows × {df.shape[1]} columns)")
                except Exception as e:
                    self.logs.append(f"⚠️ Error loading sheet '{sheet_name}': {str(e)}")
                    continue
            
            self._create_vectorstore()
            self.logs.append("✅ Vector store created successfully.")
            return "\n".join(self.logs)
        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans a DataFrame by removing empty rows/columns and converting data types."""
        df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        return df

    def _generate_sheet_description(self, sheet_name: str, df: pd.DataFrame) -> str:
        """Generates a text description of a DataFrame using an LLM."""
        sample_data = df.head(3).to_string()
        prompt = f"""
        Analyze this Excel sheet and provide a concise one-paragraph summary.
        Sheet Name: {sheet_name}
        Columns: {list(df.columns)}
        Sample Data:
        {sample_data}
        
        Focus on the main purpose of the data, key metrics, and the time period covered.
        """
        try:
            return self.llm.invoke(prompt)
        except Exception:
            return f"Sheet: {sheet_name}, Columns: {', '.join(list(df.columns))}"

    def _create_vectorstore(self):
        """Creates a FAISS vector store from sheet descriptions for similarity search."""
        documents = [
            Document(page_content=desc, metadata={"sheet_name": name})
            for name, desc in self.sheet_descriptions.items()
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

    def identify_relevant_sheets(self, query: str) -> List[str]:
        """Identifies the most relevant sheets for a given query using the vector store."""
        if not self.vectorstore:
            return list(self.excel_data.keys())
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            sheet_names = [doc.metadata['sheet_name'] for doc in docs if 'sheet_name' in doc.metadata]
            return list(dict.fromkeys(sheet_names))[:5]
        except Exception:
            return list(self.excel_data.keys())

    def query_data(self, query: str) -> Dict[str, Any]:
        """Processes a user query against the loaded Excel data."""
        results = {'query': query, 'relevant_sheets': [], 'sheet_results': {}, 'summary': '', 'insights': []}
        try:
            relevant_sheets = self.identify_relevant_sheets(query)
            results['relevant_sheets'] = relevant_sheets

            for sheet_name in relevant_sheets:
                if sheet_name not in self.excel_data:
                    continue
                df = self.excel_data[sheet_name]
                analysis_prompt = f"""
                Analyze the data from sheet '{sheet_name}' to answer the query: "{query}"
                Columns: {list(df.columns)}
                Sample Data:
                {df.head(5).to_string()}
                
                Provide a direct answer, including key numbers, trends, or patterns.
                """
                response = self.llm.invoke(analysis_prompt)
                results['sheet_results'][sheet_name] = {'response': response}
            
            results['summary'] = self._generate_summary(query, results['sheet_results'])
            results['insights'] = self._extract_insights(results['sheet_results'])
            return results
        except Exception as e:
            results['summary'] = f"Error processing query: {str(e)}"
            return results

    def _generate_summary(self, query: str, sheet_results: Dict) -> str:
        """Generates a final, consolidated summary from individual sheet analyses."""
        if not sheet_results:
            return "No relevant data found to answer the query."
        
        combined_responses = "\n\n".join(
            f"--- Analysis from Sheet '{name}' ---\n{res['response']}"
            for name, res in sheet_results.items()
        )
        prompt = f"""
        Based on the following analyses, provide a final, consolidated answer to the query.
        Original Query: {query}
        
        {combined_responses}
        
        Synthesize these findings into a clear and direct summary.
        """
        return self.llm.invoke(prompt)

    def _extract_insights(self, sheet_results: Dict) -> List[str]:
        """Extracts simple, actionable insights from the analysis results."""
        insights = set()
        for sheet_name, result in sheet_results.items():
            response = result.get('response', '').lower()
            if re.search(r'\b\d+\.?\d*\b', response):
                insights.add(f"Numerical data found in '{sheet_name}'")
            trend_keywords = ['increase', 'decrease', 'growth', 'decline', 'trend', 'pattern']
            if any(keyword in response for keyword in trend_keywords):
                insights.add(f"Trend analysis available in '{sheet_name}'")
        return list(insights)

# --- Gradio Interface ---

def process_file(api_key, file_obj):
    """Gradio function to load the file and prepare the system."""
    if not api_key:
        raise gr.Error("OpenAI API Key is required.")
    if file_obj is None:
        raise gr.Error("Please upload an Excel file.")
    try:
        excel_system = ExcelAIQuerySystem(api_key)
        loading_logs = excel_system.load_excel_file(file_obj.name)
        
        return (
            loading_logs, 
            excel_system, 
            gr.update(visible=True), 
            gr.update(visible=True),
            gr.update(visible=True)
        )
    except Exception as e:
        raise gr.Error(f"Failed to process file: {e}")

def generate_response(query, system_state):
    """Gradio function to handle user queries and display results."""
    if not query:
        raise gr.Error("Please enter a query.")
    if system_state is None:
        raise gr.Error("File not loaded. Please upload and load a file first.")
    
    try:
        result = system_state.query_data(query)
        summary = result.get('summary', 'No summary available.')
        sheets = ", ".join(result.get('relevant_sheets', []))
        insights = ", ".join(result.get('insights', []))
        
        details = f"**🔍 Relevant Sheets Identified:**\n{sheets}\n\n"
        if insights:
            details += f"**💡 Key Insights:**\n{insights}"
            
        return summary, details
    except Exception as e:
        raise gr.Error(f"Error during query: {e}")

# --- UI Layout ---

with gr.Blocks(theme=gr.themes.Soft(), title="Excel AI Query System") as demo:
    system_state = gr.State(None)

    gr.Markdown("# 📊 Excel AI Query System")
    gr.Markdown("Upload an Excel file, and ask questions about your data in plain English.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Setup")
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key...",
                value=os.getenv("OPENAI_API_KEY", "")
            )
            file_input = gr.File(label="Upload Excel File", file_types=[".xlsx", ".xls"])
            load_button = gr.Button("Load File", variant="primary")
            status_output = gr.Textbox(label="Loading Status", interactive=False, lines=5)
        
        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask a Question")
            query_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., 'What were the total sales in Q3?' or 'Show me the performance trend for Product X.'",
                visible=False
            )
            ask_button = gr.Button("Get Answer", variant="primary", visible=False)
            
            results_accordion = gr.Accordion("Results", open=False, visible=False)
            with results_accordion:
                summary_output = gr.Markdown(label="Summary")
                details_output = gr.Markdown(label="Details")

    # --- Event Handlers ---
    
    load_button.click(
        fn=process_file,
        inputs=[api_key_input, file_input],
        outputs=[status_output, system_state, query_input, ask_button, results_accordion]
    )
    
    ask_button.click(
        fn=generate_response,
        inputs=[query_input, system_state],
        outputs=[summary_output, details_output]
    ).then(
        lambda: gr.update(open=True),
        outputs=results_accordion
    )

if __name__ == "__main__":
    demo.launch(share=True)