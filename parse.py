from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

model = OllamaLLM(model="llama3").configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    # This sets a default_key.
    # If we specify this key, the default LLM (ChatAnthropic initialized above) will be used
    default_key="ollama",
    # This adds a new option, with name `gpt4omini` that is equal to `ChatOpenAI()`
    o1preview=ChatOpenAI(model="o1-preview"),    
    # This adds a new option, with name `gpt4omini` that is equal to `ChatOpenAI()`
    o1mini=ChatOpenAI(model="o1-mini"),    
    # This adds a new option, with name `gpt4omini` that is equal to `ChatOpenAI()`
    gpt4omini=ChatOpenAI(model="gpt-4o-mini"),
    # This adds a new option, with name `gpt4` that is equal to `ChatOpenAI(model="gpt-4")`
    gpt4o=ChatOpenAI(model="gpt-4o"),
    # This adds a new option, with name `google` that is equal to `ChatVertexAI(model="gemini-pro")`
    google=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    # This adds a new option, with name `mistral` that is equal to `ChatMistralAI(model="mistral-large-latest")`
    mistral=ChatMistralAI(model="mistral-large-latest")
)

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)