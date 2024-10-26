import streamlit as st
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
    remove_style_and_script_sections,
    extract_asin_info,
)

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
        # Store the DOM content in Streamlit session state
        st.session_state.dom_content = cleaned_content

        # Display the body content in an expandable text box
        with st.expander("View Body Content"):
            st.text_area("Body Content", body_content, height=300)

        # Display the body content w/o script and style sections
        with st.expander("View Body Content (no script/style)"):
            st.text_area("Body Content reduced", body_no_style_and_script, height=300)
        
        # Display the body content w/o script and style sections
        with st.expander("View product list (asin info)"):
            st.text_area("asin, href, name", asin_info, height=300)
        
        # Display the DOM content in an expandable text box
        with st.expander("View DOM Content"):
            st.text_area("DOM Content", cleaned_content, height=300)
            
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
