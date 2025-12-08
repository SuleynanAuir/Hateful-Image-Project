import os
import json
import csv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
all_data_dir = os.path.join(project_root, 'data', 'all_data')
out_dir = os.path.join(project_root, 'data', 'hateful_memes')
os.makedirs(out_dir, exist_ok=True)

in_files = [
    (os.path.join(all_data_dir, 'train.jsonl'), 'train'),
    (os.path.join(all_data_dir, 'dev.jsonl'), 'dev_seen'),
    (os.path.join(all_data_dir, 'test.jsonl'), 'test'),
]

out_csv = os.path.join(out_dir, 'hateful_memes_expanded.csv')

rows = []
for fp, split in in_files:
    if not os.path.exists(fp):
        print(f"Warning: missing {fp}, skipping")
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                'img': obj.get('img'),
                'text': obj.get('text', ''),
                'label': obj.get('label', -1),
                'split': split,
            })

with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['img', 'text', 'label', 'split'])
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Wrote {len(rows)} rows to {out_csv}")
print("Tip: use --image_pair text when running main.py")
