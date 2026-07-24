import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

options = uc.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1920,1080")
try:
    driver = uc.Chrome(options=options, version_main=150)
    driver.get("https://www.booking.com/searchresults.en-gb.html?ss=Park+Hyatt+Chicago")
    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else 'No Title'
    print(f"Title: {title}")
    if "Bot" in html or "bot" in title.lower():
        print("BLOCKED")
    else:
        print("SUCCESS")
    driver.quit()
except Exception as e:
    print(f"Error: {e}")
