from playwright.sync_api import sync_playwright
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.booking.com/searchresults.en-gb.html?ss=Park+Hyatt+Chicago")
    time.sleep(3)
    title = page.title()
    content = page.content()
    print(f"Title: {title}")
    if "Bot" in content or "bot" in title.lower():
        print("BLOCKED")
    else:
        print("SUCCESS")
    browser.close()
