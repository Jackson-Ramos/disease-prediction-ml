import json

with open('disease_prediction.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('cell_map.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(nb['cells'])}\n\n")
    for i, c in enumerate(nb['cells']):
        src = c['source']
        preview = src[0][:80].strip() if src else 'EMPTY'
        out.write(f"  Cell {i:2d}: {c['cell_type']:10s} | {preview}\n")

print("Written to cell_map.txt")
