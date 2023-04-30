import os
import json

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
    return round(len(array)*0.2)

path = "/home/student/Downloads/data/UTKFace/imgs/" #나눌 파일의 경로
file_list = []

for filename in os.listdir(path):
    file_list.append(filename)
    
male, female = [], []

for image in file_list:
    if "_0_" in image:
        male.append(image)
    else:
        female.append(image)

val_male_num = get_val_length(male)
val_female_num = get_val_length(female)

train_male_num = len(male) - val_male_num
train_female_num = len(female) - val_female_num

train_male, train_female, val_male, val_female = [], [], [], []

for i in range(train_male_num):
    train_male.append(male[i])
for i in range(train_female_num):
    train_female.append(female[i])

for j in range(val_male_num):
    val_male.append(male[j+train_male_num])
for j in range(val_female_num):
    val_female.append(female[j+train_female_num])
    

print(len(male) + len(female))
print(len(train_female) + len(train_male) + len(val_female) + len(val_male))

f = open("data/UTK-Face/gender/train.json", "w")
l = []

for i in range(train_male_num):
    d = {'img':train_male[i], 'label':'0'}
    l.append(d)
for i in range(train_female_num):
    d = {'img':train_female[i], 'label':'1'}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/gender/val.json", "w")
l = []

for i in range(val_male_num):
    d = {'img':val_male[i], 'label':'0'}
    l.append(d)
for i in range(val_female_num):
    d = {'img':val_female[i], 'label':'1'}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()