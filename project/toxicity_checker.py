import json
from memes import recognize_text, remove_text
from object_detector import get_objects
import cv2
import time
import random

start = time.time()

rel_path = '../hateful_memes/'
train_file = open(rel_path + 'train.jsonl', 'r')
train_json_list = list(train_file)
train_file.close()

results = []
for train_json in train_json_list:
    results.append(json.loads(train_json))

random.shuffle(results)

num_toxic = 0
num_good = 0
new_results = []

i = 0
N = 200
while num_toxic + num_good < N:
    if results[i]['label'] and num_toxic < N//2:
        new_results.append(results[i])
        num_toxic += 1
    if not results[i]['label'] and num_good < N//2:
        new_results.append(results[i])
        num_good += 1
    i += 1

random.shuffle(new_results)

i = 0

toxic_file = open('object_toxicity.json', 'r')
toxicity = json.loads(toxic_file.read())

correct = 0
total = 0

for img_dict in new_results:
    img_path = rel_path + img_dict['img']
    img = cv2.imread(img_path)

    img_text = recognize_text(img)
    img_text = ' '.join(img_text.split('\n'))

    img = remove_text(img)
    objects = get_objects(img)

    toxicity_score = 0
    for u,v in objects:
        toxicity_score += toxicity.get(u,0)/len(objects)

    is_toxic = toxicity_score >= 0.5
    correct += is_toxic == img_dict['label']
    if is_toxic != img_dict['label']:
        print(is_toxic, toxicity_score)
    else:
        print("<------------------------->")
    total += 1

    if i%100 == 0:
        print(f"{i} iters")
    i += 1

print()
print(f"correct: {correct}, incorrect: {total - correct}")

print(correct/total)
end = time.time()

print(end - start)