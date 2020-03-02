import json

DATASET_FILE = "News_Category_Dataset_v2.json"
cat  = []
news = []
with open(DATASET_FILE) as file:
    for d in file:
        cat.append(d)
for c in cat:
    news.append(json.loads(c))
    
news = [i for i in news if i['category'] != 'WEIRD NEWS']