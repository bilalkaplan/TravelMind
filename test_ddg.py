from duckduckgo_search import DDGS

ddgs = DDGS()
results = ddgs.text('site:booking.com "Park Hyatt Chicago"', max_results=3)
for r in results:
    print(r['title'])
    print(r['body'])
    print('-'*50)
