import json
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import arabic_reshaper
from bidi.algorithm import get_display
import time


def reshape_urdu(text):
    """Reshape Urdu text for proper display."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# Configuration
BASE_URL = "https://www.urdupoint.com/kids/section/"

# Setup Selenium with Edge browser


def setup_driver():
    options = Options()
    options.add_argument('--headless')  # Run without opening browser window
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')  # Suppress console warnings
    options.add_argument('--silent')
    options.add_argument('--disable-logging')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    options.add_experimental_option(
        'excludeSwitches', ['enable-logging', 'enable-automation'])

    # Suppress Edge driver logs
    service = Service(log_path='NUL')  # Windows null device
    driver = webdriver.Edge(service=service, options=options)
    return driver


def get_page(driver, url):
    """Fetch a webpage and return BeautifulSoup object."""
    driver.get(url)
    time.sleep(2)  # Wait for page to load
    return BeautifulSoup(driver.page_source, 'html.parser')


def extract_story(driver, url):
    """Extract story content from a page."""
    soup = get_page(driver, url)

    # Adjust selectors based on the website's HTML structure
    title = soup.find('h1').get_text(
        strip=True) if soup.find('h1') else "No title"

    # Find txt_detail container, then get the second div inside it
    txt_detail = soup.find('div', class_="txt_detail urdu ar rtl")
    text = ""

    if txt_detail:
        # Get all child divs and select the second one (index 1)
        child_divs = txt_detail.find_all('div', recursive=False)
        if len(child_divs) > 1:
            content = child_divs[1]  # Second div (index 1)
            # Get only direct text nodes, skip nested divs and p tags
            from bs4 import NavigableString
            text_parts = []
            for child in content.children:
                if isinstance(child, NavigableString):
                    stripped = child.strip()
                    if stripped:
                        text_parts.append(stripped)
            text = '\n'.join(text_parts)

    # Return original Urdu text (not reshaped) for proper storage
    return {'title': title, 'content': text, 'url': url}


def get_story_links(driver, page_url):
    """Get all story links from a listing page."""
    soup = get_page(driver, page_url)
    links = []

    # Find all story links - adjust selector as needed
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        # Filter for moral stories only
        if href and '/detail/moral-stories/' in href:
            if href.startswith('/'):
                href = 'https://www.urdupoint.com' + href
            if href not in links:
                links.append(href)

    return links


def save_stories(stories):
    """Save stories to JSON and text files."""
    # Save to JSON file
    with open('scrapper/scrapped-stories/stories.json', 'w', encoding='utf-8') as f:
        json.dump(stories, f, ensure_ascii=False, indent=2)

    # Save to formatted text file
    with open('scrapper/scrapped-stories/stories.txt', 'w', encoding='utf-8') as f:
        for i, story in enumerate(stories, 1):
            f.write(f"{'='*60}\n")
            f.write(f"Story #{i}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Title: {story['title']}\n")
            f.write(f"URL: {story['url']}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{story['content']}\n\n")


def scrape_stories(max_stories=200, num_pages=170, skip_pages=0):
    """Main scraping function."""
    driver = setup_driver()
    all_stories = []

    try:
        for page in range(1, num_pages + 1):
            if page <= skip_pages:
                print(f"Skipping page {page}")
                continue
            print(f"Scraping page {page}...")
            page_url = f"{BASE_URL}stories-page{page}.html"

            story_links = get_story_links(driver, page_url)
            print(f"  Found {len(story_links)} story links")

            for link in story_links:
                try:
                    story = extract_story(driver, link)
                    if story['content'].strip():  # Only save if content is not empty
                        all_stories.append(story)
                        print(f"  Scraped: {story['title'][:50]}...")
                    else:
                        print(
                            f"  Skipped (empty content): {story['title'][:50]}...")
                    time.sleep(1)  # Be respectful
                except Exception as e:
                    print(f"  Error scraping {link}: {e}")
                if len(all_stories) >= max_stories:
                    print(f"\nReached max stories limit ({max_stories})")
                    break

            # Save after each page
            save_stories(all_stories)
            print(f"  Saved {len(all_stories)} stories so far...")

            if len(all_stories) >= max_stories:
                break

            time.sleep(2)  # Pause between pages
    finally:
        driver.quit()  # Always close the browser

    return all_stories


# Run the scraper
stories = scrape_stories(max_stories=400, num_pages=170)
print(f"\nScraped {len(stories)} stories")
print("All stories saved!")
