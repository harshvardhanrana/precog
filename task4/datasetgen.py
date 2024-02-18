import json
import random
import csv

rel_path = '../hateful_memes/'

objects = dict()
with open('../project/all_img_objects.txt') as f:
    s = f.read().split('\n')
    for i in s:
        s2 = i.split(',')
        id = s2[0]
        objects[id] = []
        for j in s2[1:]:
            if j is not None and len(j.split(':')) == 2:
                objects[id].append(j.split(':')[0])

train_file = open(rel_path + 'train.jsonl', 'r')
train_json_list = list(train_file)
train_file.close()

results = []
for train_json in train_json_list:
    results.append(json.loads(train_json))

f = open('dataset.csv', 'w')

random.shuffle(results)

dataset = []
N = 4000
num_toxic = 0
num_good = 0
i = 0

while num_good + num_toxic < N and i < len(results):
    if results[i]['label'] and num_toxic < N//2:
        results[i]['text'] += ". "
        if len(objects[results[i]['id']]) > 0:
            results[i]['text'] += "objects: " + objects[results[i]['id']][0]
            for j in objects[results[i]['id']][1:]:
                results[i]['text'] += ", " + j
            results[i]['text'] += '.'

        dataset.append((results[i]['text'], results[i]['label']))
        num_toxic += 1
    elif not results[i]['label'] and num_good < N//2:
        results[i]['text'] += ". "
        if len(objects[results[i]['id']]) > 0:
            results[i]['text'] += "objects: " + objects[results[i]['id']][0]
            for j in objects[results[i]['id']][1:]:
                results[i]['text'] += ", " + j
            results[i]['text'] += '.'
        dataset.append((results[i]['text'], results[i]['label']))
        num_good += 1
    i += 1

random.shuffle(dataset)

with open('trainset.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text','label'])
    for i in dataset[:-800]:
        try:
            spamwriter.writerow(i)
        except:
            pass
    
with open('testset.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['text','label'])
    for i in dataset[-800:]:
        try:
            spamwriter.writerow(i)
        except:
            pass