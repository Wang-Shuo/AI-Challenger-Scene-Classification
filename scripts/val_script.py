import json
from collections import Counter
import csv

with open('scene_validation_annotations_20170908.json') as f:
	with open('val_submit_1.txt', 'r') as ff:
		preds = ff.readlines()
		labels = json.load(f)

assert len(preds) == len(labels)
errors = []
lbids = []
for pred, label in zip(preds, labels):
	pred = pred.rstrip('\n')
	if label['label_id'] not in pred:
		lbids.append(label['label_id'])
		errors.append([label['image_id'], label['label_id'], pred])

print('there are {} images predicted wrong in total {} validation images'.format(len(errors), len(labels)))

cter = Counter(lbids)

with open('id_freq.csv', 'w') as csvfile:
	field = ['label_id', 'freq']
	writer = csv.writer(csvfile)
	writer.writerow(field)
	for key, value in cter.items():
		writer.writerow([eval(key), value])


with open('error_images.txt', 'w') as f:
	for item in errors:
		f.write("%s\n" %str(item))


	
