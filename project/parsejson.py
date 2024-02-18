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
while num_toxic + num_good < 2000:
    if results[i]['label'] and num_toxic < 1000:
        new_results.append(results[i])
        num_toxic += 1
    if not results[i]['label'] and num_good < 1000:
        new_results.append(results[i])
        num_good += 1
    i += 1

random.shuffle(new_results)


tot = dict()
toxic = dict()
# f = open('img_objects_with_text.txt', 'w')
# f_text = open('img_text.txt', 'w')
i = 0

for img_dict in new_results:
    img_path = rel_path + img_dict['img']
    img = cv2.imread(img_path)

    img_text = recognize_text(img)
    img_text = ' '.join(img_text.split('\n'))
    # f_text.write(img_text)
    # f_text.write("\n")
    # print(recognize_text(img))
    # print('-------------------------')
    img = remove_text(img)
    objects = get_objects(img)
    # for u,v in objects:
    #     f.write(f"{u}:{v},")
    # f.write("\n")
    # print(objects)
    # print('-------------------------')

    for obj, v in objects:
        tot[obj] = tot.get(obj, 0) + 1

    if img_dict['label']:
        for obj, v in objects:
            toxic[obj] = toxic.get(obj, 0) + 1

    if i%100 == 0:
        print(f"{i} iters")
    i += 1

with open("toxic_new.json", "w") as outfile:
    outfile.write(json.dumps(toxic, indent=4))

with open("total_new.json", "w") as outfile:
    outfile.write(json.dumps(tot, indent=4))

# f.close()
# f_text.close()

end = time.time()

print(end - start)