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

youth, adult, elder = [], [], []

for image in file_list:
    if image[1] == "_" or int(image[0])*10 + int(image[1]) < 20:
        youth.append(image)
    elif image[2] == "_" and int(image[0])*10 + int(image[1]) >= 20 and int(image[0])*10 + int(image[1]) <= 40:
        adult.append(image)
    else:
        elder.append(image)

val_youth_num = get_val_length(youth)
train_youth_num = len(youth) - val_youth_num

val_adult_num = get_val_length(adult)
train_adult_num = len(adult) - val_adult_num

val_elder_num = get_val_length(elder)
train_elder_num = len(elder) - val_elder_num

train_youth, train_adult, train_elder, val_youth, val_adult, val_elder = [], [], [], [], [], []

for i in range(train_youth_num):
    train_youth.append(youth[i])
for i in range(train_adult_num):
    train_adult.append(adult[i])
for i in range(train_elder_num):
    train_elder.append(elder[i])
    
for j in range(val_youth_num):
    val_youth.append(youth[j+train_youth_num])
for j in range(val_adult_num):
    val_adult.append(adult[j+train_adult_num])
for j in range(val_elder_num):
    val_elder.append(elder[j+train_elder_num])

f = open("data/UTK-Face/age/train.json","w")
l = []

for i in range(train_youth_num):
    d = {'img':train_youth[i], 'label':'youth'}
    l.append(d)
for i in range(train_adult_num):
    d = {'img':train_adult[i], 'label':'adult'}
    l.append(d)
for i in range(train_elder_num):
    d = {'img':train_elder[i], 'label':'elder'}
    l.append(d)
    
json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/age/val.json","w")
l = []

for i in range(val_youth_num):
    d = {'img':val_youth[i], 'label':'youth'}
    l.append(d)
for i in range(val_adult_num):
    d = {'img':val_adult[i], 'label':'adult'}
    l.append(d)
for i in range(val_elder_num):
    d = {'img':val_elder[i], 'label':'elder'}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()