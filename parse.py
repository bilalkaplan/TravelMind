import json
try:
    with open('pyright_errors.json', 'r', encoding='utf-16le') as f:
        d = json.load(f)
except json.JSONDecodeError:
    with open('pyright_errors.json', 'r', encoding='utf-8', errors='replace') as f:
        d = json.load(f)

lines = []
for e in d['generalDiagnostics']:
    lines.append(f"{e['file']}:{e['range']['start']['line']+1} {e['message']}")

with open('pyright_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
