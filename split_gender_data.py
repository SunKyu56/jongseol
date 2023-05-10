import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_val_length(array):
    return round(len(array)*0.1)

path = "/home/student/Downloads/data/UTKFace/imgs/" #나눌 파일의 경로
file_list = []

for filename in os.listdir(path):
    file_list.append(filename)
    
male, female = [], []

for image in file_list:
    temp = image.split('_')
    if temp[1] == '0':
        male.append(image)
    else:
        female.append(image)

val_male_num = get_val_length(male)
val_female_num = get_val_length(female)

test_male_num = val_male_num
test_female_num = val_female_num

train_male_num = len(male) - val_male_num - test_male_num
train_female_num = len(female) - val_female_num - test_female_num

print(test_female_num, test_male_num)

train_male, train_female, val_male, val_female = [], [], [], []
test_male, test_female = [], []

for i in range(train_male_num):
    train_male.append(male[i])
for i in range(train_female_num):
    train_female.append(female[i])

for j in range(val_male_num):
    val_male.append(male[j+i])
for j in range(val_female_num):
    val_female.append(female[j+i])
    
for k in range(test_male_num+1):
    test_male.append(male[k+j+i])
for k in range(test_female_num+1):
    test_female.append(female[k+j+i])

print(len(male),len(female))
print(len(train_female) + len(train_male) + len(val_female) + len(val_male) + len(test_female) + len(test_male))

final_path ="data/UTK-Face/imgs/"

f = open("data/UTK-Face/gender/train.json", "w")
l = []

for i in range(train_male_num):
    d = {'img':final_path + train_male[i], 'label':0}
    l.append(d)
for i in range(train_female_num):
    d = {'img':final_path + train_female[i], 'label':1}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/gender/val.json", "w")
l = []

for i in range(val_male_num):
    d = {'img':final_path + val_male[i], 'label':0}
    l.append(d)
for i in range(val_female_num):
    d = {'img':final_path + val_female[i], 'label':1}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/gender/test.json", "w")
l = []

for i in range(test_male_num):
    d = {'img':final_path + test_male[i], 'label':0}
    l.append(d)
for i in range(test_female_num):
    d = {'img':final_path + test_female[i], 'label':1}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()