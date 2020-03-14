# %%
import json
import yaml

with open("changes.yaml", "r") as f:
    changes = yaml.load(f)
    DELETED = changes["DELETED"]
    MOVED = changes["MOVED"]

DATASET_FILE = "News_Category_Dataset_v2"
EXTENSION = "json"

with open(f"{DATASET_FILE}.{EXTENSION}", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()]

dataset = [news for news in dataset if news["category"] != "TO BE DELETED"]

dataset = [news for news in dataset if news["category"] not in DELETED]
for news in dataset:
    category = news["category"]
    if category in MOVED.values():
        news["category"] = MOVED[category]

with open(f"{DATASET_FILE}_new.{EXTENSION}", 'w') as f:
    json.dump(dataset, f, indent=4)
