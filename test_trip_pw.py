from playwright.sync_api import sync_playwright
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.trip.com/hotels/list?cityName=Chicago&keyword=Park+Hyatt+Chicago", timeout=60000)
    page.wait_for_load_state('networkidle')
    time.sleep(3)
    
    # Trip.com search results usually have links to hotels containing "/hotels/chicago-hotel-detail" or similar
    # But it's an SPA, so let's just find the first link that looks like a hotel detail
    links = page.locator("a[href*='hotel-detail']").all()
    if links:
        url = links[0].get_attribute("href")
        if url.startswith('/'):
            url = "https://www.trip.com" + url
        print(f"Found Hotel URL: {url}")
        
        page.goto(url, timeout=60000)
        page.wait_for_load_state('networkidle')
        time.sleep(5)
        
        # Room names are usually in elements with specific classes, or just look for all h2/h3
        headers = page.locator("h2, h3").all_inner_texts()
        print("Possible Rooms:")
        for h in set(headers):
            if h and len(h) > 4:
                print(f"- {h}")
    else:
        print("No hotel links found.")
        print(page.content()[:1000])
        
    browser.close()
