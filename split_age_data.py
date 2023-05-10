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

youth, student, adult, elder = [], [], [], []

for image in file_list:
    temp = image.split('_')
    if int(temp[0]) <= 15:
        youth.append(image)
    elif int(temp[0]) > 15 and int(temp[0]) <= 25:
        student.append(image)
    elif int(temp[0]) > 25 and int(temp[0]) <= 50:
        adult.append(image)
    else:
        elder.append(image)

print(len(youth),len(student),len(adult),len(elder))
print(student[1000])

val_youth_num = get_val_length(youth)
test_youth_num = val_youth_num
train_youth_num = len(youth) - 2 * val_youth_num

val_student_num = get_val_length(student)
test_student_num = val_student_num
train_student_num = len(student) - 2 * val_student_num

val_adult_num = get_val_length(adult)
test_adult_num = val_adult_num
train_adult_num = len(adult) - 2 * val_adult_num

val_elder_num = get_val_length(elder)
test_elder_num = val_elder_num
train_elder_num = len(elder) - 2 * val_elder_num

train_youth, train_adult, train_elder, train_student, val_youth, val_adult, val_elder, val_student= [], [], [], [], [], [], [], []
test_youth, test_student, test_adult, test_elder = [], [], [], []


for i in range(train_youth_num):
    train_youth.append(youth[i])
for i in range(train_student_num):
    train_student.append(student[i])
for i in range(train_adult_num):
    train_adult.append(adult[i])
for i in range(train_elder_num):
    train_elder.append(elder[i])
    
for j in range(val_youth_num):
    val_youth.append(youth[j+train_youth_num])
for j in range(val_student_num):
    val_student.append(student[j+train_student_num])
for j in range(val_adult_num):
    val_adult.append(adult[j+train_adult_num])
for j in range(val_elder_num):
    val_elder.append(elder[j+train_elder_num])

for k in range(test_youth_num):
    test_youth.append(youth[k+train_youth_num+val_youth_num])
for k in range(test_student_num):
    test_student.append(student[k+train_student_num+val_student_num])
for k in range(test_adult_num):
    test_adult.append(adult[k+train_adult_num+val_adult_num])
for k in range(test_elder_num):
    test_elder.append(elder[k+train_elder_num+val_elder_num])

final_path = "data/UTK-Face/imgs/"

f = open("data/UTK-Face/age/train.json","w")
l = []

for i in range(train_youth_num):
    d = {'img':final_path + train_youth[i], 'label':0}
    l.append(d)
for i in range(train_student_num):
    d = {'img':final_path + train_student[i], 'label':1}
    l.append(d)
for i in range(train_adult_num):
    d = {'img':final_path + train_adult[i], 'label':2}
    l.append(d)
for i in range(train_elder_num):
    d = {'img':final_path + train_elder[i], 'label':3}
    l.append(d)
    
json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/age/val.json","w")
l = []

for i in range(val_youth_num):
    d = {'img':final_path + val_youth[i], 'label':0}
    l.append(d)
for i in range(val_student_num):
    d = {'img':final_path + val_student[i], 'label':1}
    l.append(d)
for i in range(val_adult_num):
    d = {'img':final_path + val_adult[i], 'label':2}
    l.append(d)
for i in range(val_elder_num):
    d = {'img':final_path + val_elder[i], 'label':3}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()

f = open("data/UTK-Face/age/test.json","w")
l = []

for i in range(val_youth_num):
    d = {'img':final_path + test_youth[i], 'label':0}
    l.append(d)
for i in range(val_student_num):
    d = {'img':final_path + test_student[i], 'label':1}
    l.append(d)
for i in range(val_adult_num):
    d = {'img':final_path + test_adult[i], 'label':2}
    l.append(d)
for i in range(val_elder_num):
    d = {'img':final_path + test_elder[i], 'label':3}
    l.append(d)

json.dump(l,f,indent='\t',cls=NpEncoder)
f.close()