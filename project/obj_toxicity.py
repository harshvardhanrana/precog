import json

f = open('total_new.json')

f2 = open('toxic_new.json')

total = json.loads(f.read())

toxic = json.loads(f2.read())

toxicity_score = dict()

for u,v in total.items():
    if total[u] <= 4:
        continue
    toxicity_score[u] = toxic.get(u,0)/total[u]

with open("object_toxicity.json", "w") as outfile:
    outfile.write(json.dumps(toxicity_score, indent=4))