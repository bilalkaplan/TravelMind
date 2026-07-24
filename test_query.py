import sys
sys.path.append('src')
try:
    from cmu_rag_answer import rewrite_search_query
    hist = [{'role': 'user', 'content': 'Chicago da luks otel arıyorum'}, {'role': 'assistant', 'content': 'Tabii, birkaç otel var.'}]
    res = rewrite_search_query('peki bunlardan havuzu olan var mı?', hist, 'Chicago, IL')
    print('REWRITTEN:', res)
except Exception as e:
    print('ERROR:', e)
