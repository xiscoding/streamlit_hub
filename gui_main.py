import os
import streamlit as st
from dotenv import load_dotenv, set_key
import getpass
from run_model import model_selector, run_parser, template1, template2

class EnvConfig:
    def __init__(self):
        load_dotenv()
        self.huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.openai_api_token = self._set_if_undefined("OPENAI_API_KEY")
        self.google_genai_api_token = self._set_if_undefined("GOOGLE_GENAI_API")

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
    
    def setup_llm(self, task: str, model_name: str):
        model = model_selector(model_name)
        return run_parser(parse_description=task, template=template1)

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
                self.models[model_name] = QueryExecutor(setup_instance.setup_llm(task="", model_name=model_name))
            except Exception as e:
                print(f"Error setting up model {model_name}: {e}")

    def list_models(self):
        return list(self.models.keys())

    def get_model(self, model_name):
        return self.models[model_name]

class QueryExecutor:
    def __init__(self, model):
        self.model = model

    def ask_question(self, question: str, template: str, task: str):
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

class StreamlitApp:
    def __init__(self):
        load_dotenv()
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

    def display_messages(self):
        message_container = st.container()
        with message_container:
            st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
            for role, message in st.session_state.messages:
                if role == "You":
                    st.markdown(f"<div class='message-right'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='message-left'>{message}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        self.setup_page_layout()
        st.sidebar.title("Choose your Language Model")
        model_options = self.model_selector.list_models()
        chosen_model = st.sidebar.selectbox("Select a model", model_options)

        # Define task-template pairs
        task_template_pairs = {
            "Summarization": "template1",
            "Question Answering": "template2",
            "Sentiment Analysis": "template3"
        }

        st.sidebar.title("Choose a Template Task")
        task_options = list(task_template_pairs.keys())
        chosen_task = st.sidebar.selectbox("Select a task", task_options)
        chosen_template = task_template_pairs[chosen_task]

        st.title("Ask Your Model")
        question = st.text_input("Enter your question:")

        if st.button("Submit"):
            if question:
                query_executor = self.model_selector.get_model(chosen_model)
                output = query_executor.ask_question(question=question, template=chosen_template, task=chosen_task)
                st.write(output)

if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
