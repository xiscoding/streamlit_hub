from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM 

template1 = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

template2 = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

def model_selector(model:str):
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

def create_chain(template:str, model):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model #| StrOutputParser()
    return chain

def run_parser(chain, parse_description:str, content:str=None, chunked_data:list=None):
    if chunked_data:
        parsed_results = []
        for i, chunk in enumerate(chunked_data, start=1):
            response = chain.invoke(
                {"dom_content": chunk, "parse_description": parse_description}
            )
            print(f"Chunk {i}: {response}")
            parsed_results.append(response)
        return "\n".join(parsed_results)
    
    response = chain.invoke(
        {"dom_content": content, "parse_description": parse_description}
    )
    return response

if __name__ == '__main__':
    from scrape import scrape_website, extract_body_content
    from dotenv import load_dotenv
    import os

    load_dotenv()
    openai_api_token = os.getenv("OPENAI_API_KEY")
    model = model_selector("gpt-4o-mini")
    parse_description = "Examples"
    url = "https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html"
    raw_content = scrape_website(url)
    content = extract_body_content(raw_content)
    results = run_parser(parse_description=parse_description, template=template1, content=content, model=model)
    print(results)


    
    