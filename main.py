import os
import streamlit as st
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
    remove_style_and_script_sections,
    extract_asin_info,
    save_content
)

SAVE_DIR = '/home/xdoestech/Desktop/amazon_scraper/saved_files'

# Streamlit UI
st.title("AI Web Scraper")
url = st.text_input("Enter Website URL")

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
            save_content(os.path.join(SAVE_DIR, "raw_html_body"), data["body_content"], "Body Content")

    with st.expander("View Body Content (no script/style)"):
        st.text_area("Body Content Reduced", data["body_no_style_and_script"], height=300)
        if st.button("Save Reduced Body Content", key="save_reduced_body"):
            save_content(os.path.join(SAVE_DIR, "html_body_no_style_script"), data["body_no_style_and_script"], "Reduced Body Content")

    with st.expander("View Product List (ASIN Info)"):
        st.text_area("ASIN Info", data["asin_info"], height=300)
        if st.button("Save ASIN Info", key="save_asin_info"):
            save_content(os.path.join(SAVE_DIR, "extracted_asin_info"), data["asin_info"], "ASIN Info")

    with st.expander("View DOM Content"):
        st.text_area("DOM Content", data["dom_content"], height=300)
        if st.button("Save DOM Content", key="save_dom_content"):
            save_content(os.path.join(SAVE_DIR, "extracted_dom_content"), data["dom_content"], "DOM Content")
            
# Step 2: Ask Questions About the DOM Content
if "dom_content" in st.session_state:
    parse_description = st.text_area("Describe what you want to parse")

    if st.button("Parse Content"):
        if parse_description:
            st.write("Parsing the content...")

            # Parse the content with Ollama
            dom_chunks = split_dom_content(st.session_state.dom_content)
            parsed_result = parse_with_ollama(dom_chunks, parse_description)
            st.write(parsed_result)
