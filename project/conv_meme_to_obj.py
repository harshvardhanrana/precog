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

f = open('all_img_objects.txt', 'w')
i = 0

for img_dict in results:
    img_path = rel_path + img_dict['img']
    img = cv2.imread(img_path)

    img_text = recognize_text(img)
    img_text = ' '.join(img_text.split('\n'))
    # f_text.write(img_text)
    # f_text.write("\n")

    img = remove_text(img)
    objects = get_objects(img)
    f.write(f"{img_dict['id']},")
    for u,v in objects:
        f.write(f"{u}:{v},")
    f.write("\n")

    if i%50 == 0:
        print(f"{i} iters")
        f.flush()
    i += 1

f.close()

end = time.time()

print(end - start)