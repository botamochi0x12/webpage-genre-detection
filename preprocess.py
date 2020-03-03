import json

DATASET_FILE = "News_Category_Dataset_v2"
EXTENSION = ".json"
cat  = []
news = []
with open(DATASET_FILE + EXTENSION) as file:
    for d in file:
        cat.append(d)
for c in cat:
    news.append(json.loads(c))    

news = [i for i in news if i['category'] != 'TO BE DELETED']

with open(DATASET_FILE + "_new" + EXTENSION, 'w') as file:
    json.dump(news, file, indent=4)