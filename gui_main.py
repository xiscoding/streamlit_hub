import os
import streamlit as st
from dotenv import load_dotenv, set_key
import getpass
from run_model import  template1, template2
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
    remove_style_and_script_sections,
    extract_asin_info,
    save_text_to_file,
    create_unique_filename
)
import time

SAVE_DIR = '/home/xdoestech/Desktop/amazon_scraper/saved_files'

class EnvConfig:
    def __init__(self):
        self.huggingfacehub_api_token = self._set_if_undefined("HUGGINGFACE_API_TOKEN")
        self.openai_api_token = self._set_if_undefined("OPENAI_API_KEY")
        self.google_genai_api_token = self._set_if_undefined("GOOGLE_API_KEY")

    def _set_if_undefined(self, var: str):
        if not os.environ.get(var):
            value = getpass.getpass(f"Please provide your {var}")
            os.environ[var] = value
            # Save to .env file
            set_key('.env', var, value)
        return os.environ.get(var)
    
class ModelSetup:
    def __init__(self, api_token):
        self.api_token = api_token

    def model_selector(self, model:str):
        if model == "ollama":
            return OllamaLLM(model="llama3")
        if model == "mistral-large-latest":
            return ChatMistralAI(model="mistral-large-latest")
        if model == "gemini-1.5-pro":
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        if model == "gpt-4o":
            return ChatOpenAI(model="gpt-4o")
        if model == "gpt-4o-mini":
            return ChatOpenAI(model="gpt-4o-mini")
        if model == "o1-preview":
            return ChatOpenAI(model="o1-preview")
        if model == "o1-mini":
            return ChatOpenAI(model="o1-mini")

    def setup_llm(self, model_name: str):
        model = self.model_selector(model_name)
        return model
    
class ModelSelector:
    def __init__(self, env_config):
        self.cloud_model_setup = ModelSetup(env_config.huggingfacehub_api_token)
        self.local_model_setup = ModelSetup(None)  # API token not needed for local models
        self.openai_model_setup = ModelSetup(env_config.openai_api_token)
        self.google_model_setup = ModelSetup(env_config.google_genai_api_token)
        
        # Dictionary to map model names to their respective setup configurations
        self.model_configurations = {
            "ollama": (self.local_model_setup, "ollama"),
            "mistral-large-latest": (self.cloud_model_setup, "mistral-large-latest"),
            "gemini-1.5-pro": (self.cloud_model_setup, "gemini-1.5-pro"),
            "gpt-4o": (self.openai_model_setup, "gpt-4o"),
            "gpt-4o-mini": (self.openai_model_setup, "gpt-4o-mini"),
            "o1-preview": (self.openai_model_setup, "o1-preview"),
            "o1-mini": (self.openai_model_setup, "o1-mini")
        }

        # Dynamically create the models dictionary
        self.models = {}
        for model_name, config in self.model_configurations.items():
            setup_instance, model_name = config
            try:
                self.models[model_name] = setup_instance.setup_llm(model_name=model_name)
            except Exception as e:
                print(f"Error setting up model {model_name}: {e}")

    def list_models(self):
        return list(self.models.keys())

    def get_model(self, model_name):
        return self.models[model_name]

class QueryExecutor:
    def __init__(self, model):
        self.model = model
        
    def create_chain(template:str, model):
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model #| StrOutputParser()
        return chain

    def ask_question(self, question: str, task: str, template: str=None):
        if (template):
            chain = self.create_chain(template, self.model)
            chain.invoke(question)
        else:
            result = self.model.invoke(question)
        # Ensure the output is structured as expected
        if isinstance(result, dict):
            output = {'Question': question, 'Answer': result.get('text', "No answer found.")}
        elif isinstance(result, str):
            output = {'Question': question, 'Answer': result}
        else:
            try:
                output = {'Question': question, 'Answer': result.content}
            except AttributeError as e:
                output = {'Question': question, 'Answer': f"AttributeError: {str(e)}"}
            except Exception as e:
                output = {'Question': question, 'Answer': f"Unexpected error: {str(e)}"}
        return output

class FileSaver:
    @staticmethod
    def save_to_file(file_path, content):
        file_path = os.path.join(SAVE_DIR, file_path)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    
    @staticmethod
    def create_unique_filename(base_name, extension='.txt'):
        """
        Create a unique file name by appending the current timestamp to the base name.

        :param base_name: The base name of the file (without extension).
        :param extension: The file extension (e.g., '.txt').
        :return: A unique file name.
        """
        # if not base_name:
        #     raise ValueError("Base name cannot be empty.")
        # if not extension:
        #     raise ValueError("Extension cannot be empty.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        unique_filename = f"{base_name}_{timestamp}{extension}"

        # Ensure the file name is unique by checking if it already exists
        counter = 1
        while os.path.exists(unique_filename):
            unique_filename = f"{base_name}_{timestamp}_{counter}{extension}"
            counter += 1

        return unique_filename 

class StreamlitApp:
    def __init__(self):
        self.env_config = EnvConfig()
        self.model_selector = ModelSelector(self.env_config)
        self.chat_manager = None

    def setup_page_layout(self):
        st.markdown("""
            <style>
            .message-left {
                text-align: left;
                color: #FFA500;
            }
            .message-right {
                text-align: right;
                color: #00A5FF;
            }
            .chat-messages {
                height: 450px;
                overflow-y: auto;
            }
            </style>
            """, unsafe_allow_html=True)

    def update_ui_for_task(self, chosen_task):
        """Update UI components dynamically based on the selected task."""
        task_prompts = {
            "No template": "Enter your question:",
            "Summarization": "Enter text to summarize:",
            "Question Answering": "Ask a question about the given context:",
            "Language Learning": "Enter a sentence to analyze sentiment:",
            "Web Scraping": None  # Web scraping has its own custom UI logic
        }
        return task_prompts.get(chosen_task, "Enter your question:")

    def display_web_scraping_ui(self):
        """Custom UI for Web Scraping task with save functionality."""
        st.title("AI Web Scraper")
        
        # Input URL
        url = st.text_input("Enter Website URL", key="web_scraping_url")

        # Step 1: Scrape the Website
        if st.button("Scrape Website"):
            if url:
                st.write("Scraping the website...")

                # Scrape the website
                dom_content = scrape_website(url)
                body_content = extract_body_content(dom_content)
                cleaned_content = clean_body_content(body_content)
                body_no_style_and_script = remove_style_and_script_sections(dom_content)
                asin_info = extract_asin_info(dom_content)

                # Store all data in session state
                st.session_state["web_scraping_data"] = {
                    "dom_content": cleaned_content,
                    "body_content": body_content,
                    "body_no_style_and_script": body_no_style_and_script,
                    "asin_info": asin_info,
                }
                st.success("Website scraped successfully!")

        # Ensure scraping data exists in session state
        if "web_scraping_data" in st.session_state:
            data = st.session_state["web_scraping_data"]

            # Display sections for content
            with st.expander("View Body Content"):
                st.text_area("Body Content", data["body_content"], height=300)
                if st.button("Save Body Content", key="save_body_content"):
                    self.save_content("raw_html_body", data["body_content"], "Body Content")

            with st.expander("View Body Content (no script/style)"):
                st.text_area("Body Content Reduced", data["body_no_style_and_script"], height=300)
                if st.button("Save Reduced Body Content", key="save_reduced_body"):
                    self.save_content("html_body_no_style_script", data["body_no_style_and_script"], "Reduced Body Content")

            with st.expander("View Product List (ASIN Info)"):
                st.text_area("ASIN Info", data["asin_info"], height=300)
                if st.button("Save ASIN Info", key="save_asin_info"):
                    self.save_content("extracted_asin_info", data["asin_info"], "ASIN Info")

            with st.expander("View DOM Content"):
                st.text_area("DOM Content", data["dom_content"], height=300)
                if st.button("Save DOM Content", key="save_dom_content"):
                    self.save_content("extracted_dom_content", data["dom_content"], "DOM Content")

    def save_content(self, base_name, content, content_type):
        """Save content to a file."""
        try:
            file_path = FileSaver.create_unique_filename(base_name)
            FileSaver.save_to_file(file_path, content)
            st.success(f"{content_type} saved successfully as {file_path}!")
        except Exception as e:
            st.error(f"Error saving {content_type}: {str(e)}")

    def run(self):
        self.setup_page_layout()
        st.sidebar.title("Choose your Language Model")
        model_options = self.model_selector.list_models()
        chosen_model = st.sidebar.selectbox("Select a model", model_options)

        # Define task-template pairs
        task_template_pairs = {
            "No template": None,
            "Summarization": "template1",
            "Question Answering": "template2",
            "Language Learning": "template3",
            "Web Scraping": "web_scraping"
        }

        st.sidebar.title("Choose a Template Task")
        task_options = list(task_template_pairs.keys())
        chosen_task = st.sidebar.selectbox("Select a task", task_options)

        # Check if chosen task is Web Scraping and render its custom UI
        if chosen_task == "Web Scraping":
            self.display_web_scraping_ui()
        else:
            # Render generic task UI for other templates
            input_label = self.update_ui_for_task(chosen_task)
            question = st.text_input(input_label)
            st.write(f"Chosen Template: {task_template_pairs[chosen_task]}")

            if st.button("Submit"):
                if question:
                    query_executor = QueryExecutor(self.model_selector.get_model(chosen_model))
                    output = query_executor.ask_question(
                        question=question,
                        template=task_template_pairs[chosen_task],
                        task=chosen_task
                    )
                    st.write(output)



if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
