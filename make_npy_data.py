# dcoding=utf-8

import os
from PIL import Image
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

with open('impression_scores/impression_scores_wordtop10_soph_1936_regress_tag.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    imp = np.array([])
    for row in reader:
        tag = np.array([])
        ro = row[2]
        imp = np.append(imp,float(ro)).reshape(-1,1)

np.save('np_array/impression_score_1936_soph_kmean7',imp)

print(imp)
print(imp.shape)

exit()

with open('impression_scores/impression_scores_wordtop10_soph_1936_regress_tag.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    imp = np.array([])

    for row in reader:
        ro = row[3:]
        tag = np.array([])
        for r in ro:
            tag = np.append(tag, float(r))
        imp = np.append(imp, tag).reshape(-1, 7)

np.save('np_array/impression_score_1936_soph_regress',imp)

print(imp)
print(imp.shape)

exit()


impression = np.array([])

with open('impression_scores/impression_scores_wordtop10_soph_19360_1to7.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    list = []

    #impression = np.array([])
    watch_images = np.array([])
    n = 0
    for row in tqdm(reader):
        #print(row)
        imp = np.array([])

        list.append(row[0])

        im = np.array(Image.open('all_images_id/{}'.format(row[0])),'f')/255.0

        watch_images = np.append(watch_images, im).reshape(-1,128,128,3)

        class7 = row[3:]
        #print(class7)
        for c in class7:
            imp = np.append(imp, float(c))

        #print(imp)

        impression = np.append(impression, imp).reshape(-1,7)

        #print(impression)
        #exit()

    #np.save('np_array/watch_images_array_19360_ZtoA_set{}'.format(i+1),watch_images)
    #print(watch_images.shape)

    #np.save('np_array/impression_label_1936_soph_kmean7_set{}'.format(i+1), impression)
    #print(impression.shape)

np.save('np_array/impression_label_19360_soph_kmean7_Norand', impression)
print(impression.shape)

exit()

impression = np.array([])

with open('impression_scores/impression_scores_wordtop10_soph_19360_1to7.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    list = []

    #impression = np.array([])
    watch_images = np.array([])
    n = 0
    for row in tqdm(reader):
        #print(row)
        imp = np.array([])

        list.append(row[0])

        #im = np.array(Image.open('all_images_id/{}'.format(row[0])),'f')/255.0

        #watch_images = np.append(watch_images, im).reshape(-1,128,128,3)

        class7 = row[3:]
        #print(class7)
        for c in class7:
            imp = np.append(imp, float(c))

        #print(imp)

        impression = np.append(impression, imp).reshape(-1,7)

        #print(impression)
        #exit()

    #np.save('np_array/watch_images_array_19360_ZtoA_set{}'.format(i+1),watch_images)
    #print(watch_images.shape)

    #np.save('np_array/impression_label_1936_soph_kmean7_set{}'.format(i+1), impression)
    #print(impression.shape)

np.save('np_array/impression_label_19360_soph_kmean7_Norand', impression)
print(impression.shape)


with open('impression_scores/impression_scores_wordtop10_rich_1936_tag.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    imp = np.array([])
    for row in reader:
        tag = np.array([])
        ro = row[1:]
        for r in ro:
            tag=np.append(tag, float(r))
        imp = np.append(imp,tag).reshape(-1,7)

np.save('np_array/impression_label_1936_rich_kmean7',imp)

print(imp)
print(imp.shape)

exit()

with open('impression_scores/impression_scores_wordtop10_soph_1936_1to7.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    watch_images = np.array([])
    imp = np.array([])
    list = []
    n = 0
    for row in reader:

        list.append(row[0])

        score = row[1:]

        for r in score:

            imp = np.append(imp,[int(r)]).reshape(-1,1)

np.save('np_array/impression_score_array_19360_soph_1to7',imp)
print(imp.shape)

df = pd.DataFrame(list)
df.to_csv("np_array/image_list_1936.csv", header = False, index = False)

print(watch_images.shape)
print(imp.shape)


with open('impression_scores/id_score_text_crowd_wordtop10_rich_1936.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    watch_images = np.array([])
    imp = np.array([])
    list = []
    n = 0
    for row in reader:

        list.append(row[0])

        im = np.array(Image.open('all_images_id/{}'.format(row[0])),'f')/255.0

        watch_images = np.append(watch_images, im).reshape(-1,128,128,3)

np.save('np_array/watch_images_array_1936_ZtoA',watch_images)
print(watch_images.shape)
exit()

with open('id_num_score_text_crowd_1950_sortscore.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    imp = np.array([])
    k = np.array([])

    for row in reader:
        k = np.append(k,[float(row[9])]).reshape(-1,1)

pred = KMeans(n_clusters=7).fit_predict(k)
print(pred)

pred_sort = np.array([])
now = pred[0]
num = 1

for p in pred:
    if p != now:
        now = p
        num += 1

    pred_sort = np.append(pred_sort, num)


imp = pred_sort.reshape(-1,1)

print(imp)
print(len(imp))

df = pd.DataFrame(imp)
#df.to_csv("impression_score_array_1950_rich_kmean7.csv", header = False, index = False)

with open('id_num_score_text_crowd_1950.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    imp = np.array([])
    for row in reader:
        tag = np.array([0,0,0,0,0,0,0])
        tag[int(row[11])-1] = 1
        imp = np.append(imp,tag).reshape(-1,7)


np.save('np_array/impression_score_array_1950_soph_kmean7',imp)

print(imp)
print(imp.shape)

with open('id_num_score_text_crowd_1950.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    watch_images = np.array([])
    imp = np.array([])
    list = []
    n = 0
    for row in reader:

        list.append(row[0])

        #im = np.array(Image.open('all_images_id/{}'.format(row[0])),'f')/255.0

        #watch_images = np.append(watch_images, im).reshape(-1,128,128,3)

        imp = np.append(imp,[float(row[9])]).reshape(-1,1)

#np.save('np_array/watch_images_array_1950_ZtoA',watch_images)
np.save('np_array/impression_score_array_1950_rich_1to7',imp)

df = pd.DataFrame(list)
df.to_csv("np_array/image_list_1950.csv", header = False, index = False)

print(watch_images.shape)
print(imp.shape)

exit()


with open('test_50.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    watch_images = np.array([])
    imp = np.array([])
    n = 0
    for row in reader:

        im = np.array(Image.open('test_images_50/{}'.format(row[0])),'f')/255.0

        watch_images = np.append(watch_images, im).reshape(-1,128,128,3)

        imp = np.append(imp,[float(row[3])]).reshape(-1,1)

np.save('np_array/watch_images_array_test_50',watch_images)
np.save('np_array/impression_score_array_test_50',imp)

print(watch_images.shape)
print(imp.shape)

B = 10000000.0

with open('features_test_50/GramMat64.csv', 'r') as f:
    reader = csv.reader(f)
    gram_64_array = np.array([])
    n = 0
    for row in reader:
        gram_64_array = np.append(gram_64_array, [float(n) for n in row]).reshape(-1,64,64,1)/B

print(gram_64_array.shape)
np.save('np_array/gram_64_array_test_50',gram_64_array)

with open('features_test_50/GramMat128.csv', 'r') as f:
    reader = csv.reader(f)
    gram_128_array = np.array([])
    n = 0
    for row in reader:
        gram_128_array = np.append(gram_128_array, [float(n) for n in row]).reshape(-1,128,128,1)/B

print(gram_128_array.shape)
np.save('np_array/gram_128_array_test_50',gram_128_array)

with open('features/GramMat256.csv', 'r') as f:
    reader = csv.reader(f)
    gram_256_array = np.array([])
    n = 0
    for row in reader:
        gram_256_array = np.append(gram_256_array, [float(n) for n in row]).reshape(-1,256,256,1)/B

np.save('np_array/gram_256_array',gram_256_array)
print(gram_256_array.shape)

with open('features_test_50/GramMat512.csv', 'r') as f:
    reader = csv.reader(f)
    gram_512_array = np.array([])
    n = 0
    for row in reader:
        gram_512_array = np.append(gram_512_array, [float(n) for n in row]).reshape(-1,512,512,1)/B

print(gram_512_array.shape)
np.save('np_array/gram_512_array_test_50',gram_512_array)
