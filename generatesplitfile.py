import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#dataset_dir = '/Developer/Dataset/Argoverse/argoverse-conv-rect-mynew/training/'#only train1, 1903 images
dataset_dir = '/Developer/Dataset/Argoverse/argoverse-conv-rect-all/training/'#all folders, 9100 images

import os
image_2 = 'image_2/'
images = os.listdir(dataset_dir + image_2)
print(len([img for img in images]))

dataset = []
for img in images:
    dataset.append(img[:-4])#remove .png

df = pd.DataFrame(dataset, columns=['index'], dtype=np.int32)

X_trainval, X_test = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)
X_train, X_subval = train_test_split(X_trainval, train_size=0.75, test_size=0.25, random_state=42)
X_trainval.shape, X_train.shape, X_subval.shape, X_test.shape

X_test.sort_values('index')

def write_to_file(path, data): 
    file = open(path, 'w') 
    for idx in data: 
        #print(idx)
        file.write(str(idx).zfill(6))
        file.write('\n')

    file.close()
    print('Done in ' + path)

write_to_file('./split/fullargo.txt', df.sort_values('index')['index'])
write_to_file('./split/trainval_fullargo.txt', X_trainval.sort_values('index')['index'])
write_to_file('./split/train_fullargo.txt', X_train.sort_values('index')['index'])
write_to_file('./split/subval_fullargo.txt', X_subval.sort_values('index')['index'])
write_to_file('./split/test_fullargo.txt', X_test.sort_values('index')['index'])