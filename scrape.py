import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

def scrape_website(website):
    """
    Scrape the HTML content of the given website.

    :param website: The URL of the website to scrape.
    :return: The HTML content of the page.
    """
    if not website:
        raise ValueError("Website URL cannot be empty. Please provide a valid URL.")

    print("Launching browser...")

    # Uncomment and update the path if using Chrome
    # chrome_driver_path = "<YOUR_CHROME_DRIVER_PATH>"
    # options = webdriver.ChromeOptions()
    # driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    # Using Edge driver as a placeholder
    driver = webdriver.Edge()  # Ensure Edge WebDriver is installed and available

    try:
        print(f"Loading page: {website}")
        driver.get(website)
        print("Page loaded...")
        time.sleep(10)  # Wait to ensure page is fully loaded; adjust as needed
        html = driver.page_source
        return html
    except Exception as e:
        print(f"Error occurred while loading the page: {e}")
        raise
    finally:
        driver.quit()
        print("Browser closed.")

def extract_body_content(html_content):
    """
    Extract the body content from the HTML content.

    :param html_content: The full HTML content of the page.
    :return: The body content as a string.
    """
    if not html_content:
        raise ValueError("HTML content cannot be empty.")

    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    else:
        print("Warning: No body tag found in the HTML content.")
    return ""

def extract_div_sections(html_content):
    """
    Extract all div sections with the attribute 'data-asin'.

    :param html_content: The full HTML content of the page.
    :return: A list of div sections with 'data-asin' attribute.
    """
    if not html_content:
        raise ValueError("HTML content cannot be empty.")

    soup = BeautifulSoup(html_content, "html.parser")
    div_sections = soup.find_all('div', attrs={'data-asin': True})
    return [str(div) for div in div_sections]

def extract_asin_info(html_content):
    """
    Extract 'asin', 'href', and product name from all div sections with 'data-asin'.

    :param html_content: The full HTML content of the page.
    :return: A list of dictionaries containing 'asin', 'href', and 'name'.
    """
    if not html_content:
        raise ValueError("HTML content cannot be empty.")

    soup = BeautifulSoup(html_content, "html.parser")
    div_sections = soup.find_all('div', attrs={'data-asin': True})
    asin_info_list = []

    for div in div_sections:
        asin = div.get('data-asin')
        link_tag = div.find('a', class_='a-link-normal', href=True)
        product_name_tag = div.find('span', class_='a-size-base-plus')

        href = link_tag['href'] if link_tag else None
        product_name = product_name_tag.text.strip() if product_name_tag else None

        asin_info = {
            'asin': asin,
            'href': href,
            'name': product_name
        }
        asin_info_list.append(asin_info)

    return asin_info_list

def clean_body_content(body_content):
    """
    Clean the body content by removing script and style tags and extracting visible text.

    :param body_content: The HTML body content.
    :return: Cleaned text content.
    """
    if not body_content:
        raise ValueError("Body content cannot be empty.")

    soup = BeautifulSoup(body_content, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Extract and clean text
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content

def remove_style_and_script_sections(html_content):
    """
    Remove all style and script sections from the HTML content.

    :param html_content: The full HTML content of the page.
    :return: The HTML content without style and script sections.
    """
    if not html_content:
        raise ValueError("HTML content cannot be empty.")

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    return str(soup)

def split_dom_content(dom_content, max_length=6000):
    """
    Split the DOM content into smaller chunks based on the specified max length.

    :param dom_content: The cleaned DOM content.
    :param max_length: The maximum length of each chunk.
    :return: A list of content chunks.
    """
    if not dom_content:
        raise ValueError("DOM content cannot be empty.")
    
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]

def save_text_to_file(text, filename):
    """
    Save the provided text to a file.

    :param text: The text to save.
    :param filename: The name of the file where the text will be saved.
    """
    if not text:
        raise ValueError("Text content cannot be empty.")
    if not filename:
        raise ValueError("Filename cannot be empty.")

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text successfully saved to {filename}.")

def create_unique_filename(base_name, extension):
    """
    Create a unique file name by appending the current timestamp to the base name.

    :param base_name: The base name of the file (without extension).
    :param extension: The file extension (e.g., '.txt').
    :return: A unique file name.
    """
    if not base_name:
        raise ValueError("Base name cannot be empty.")
    if not extension:
        raise ValueError("Extension cannot be empty.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    unique_filename = f"{base_name}_{timestamp}{extension}"

    # Ensure the file name is unique by checking if it already exists
    counter = 1
    while os.path.exists(unique_filename):
        unique_filename = f"{base_name}_{timestamp}_{counter}{extension}"
        counter += 1

    return unique_filename

if __name__ == '__main__':
    load_dotenv()  # Load environment variables from a .env file if needed
    
    url = os.getenv('SCRAPE_URL')  # Fetch URL from environment variable
    if not url:
        print("Error: No URL provided. Please set the SCRAPE_URL environment variable.")
    else:
        try:
            result = scrape_website(url)
            body_content = extract_body_content(result)
            div_sections = extract_div_sections(result)
            
            for i, div in enumerate(div_sections):
                filename = create_unique_filename(f"scraped_div_section_{i+1}", ".txt")
                save_text_to_file(div, filename)
                print(f"Div section {i+1} saved to {filename}.\n")
            
            cleaned_content = clean_body_content(body_content)
            split_content = split_dom_content(cleaned_content)
            
            for i, content in enumerate(split_content):
                filename = create_unique_filename(f"scraped_content_part_{i+1}", ".txt")
                save_text_to_file(content, filename)
                print(f"Content part {i+1} saved to {filename}.\n")
            
            # Extract ASIN information and print
            asin_info_list = extract_asin_info(result)
            for info in asin_info_list:
                print(info)
        except Exception as e:
            print(f"An error occurred: {e}")
