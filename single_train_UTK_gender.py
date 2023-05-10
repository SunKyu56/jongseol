import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from data.UTKDataset2 import UTKDataset
import numpy as np
from tqdm import tqdm 
from torchvision.transforms import transforms
from make_json import NpEncoder
import json
import wandb
import cv2, time, copy
import torch.optim.lr_scheduler as lr_scheduler

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

########
wandb.init(project="gender", entity="kookmin_ai")
wandb.run.name = "resnet_18_gender_Adam1"
########

device='cuda:0' if torch.cuda.is_available() else 'cpu'

train_dataset = UTKDataset(json_path="data/UTK-Face/gender/train.json")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

val_dataset = UTKDataset(json_path="data/UTK-Face/gender/val.json")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define hyperparameters
num_epochs = 100

# Create ResNet18 model
model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Replace last fully connected layer for binary classification
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#saving best loss model
best_loss = float('inf')

saving_log = []

# Train model
for epoch in range(num_epochs):
    current_lr = get_lr(optimizer)
    running_loss = 0.0
    model.train()
    running_total = 0
    running_correct = 0
    for i, data in enumerate(tqdm(train_loader, 0)):
        inputs, label = data
        labels = torch.as_tensor(np.array(label).astype(int))
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
    running_loss /= len(train_loader)
    running_accuracy = 100 * running_correct/ running_total
            
    val_loss = 0
    val_total = 0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, 0)):
            inputs, label = data
            labels = torch.as_tensor(np.array(label).astype(int))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct/ val_total
    
    wandb.log({'train_loss':running_loss,
               'val_loss':val_loss,
               'train_accuracy':running_accuracy,
               'val_accuracy':val_accuracy
               },step=epoch+1)
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_gender_acc = val_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "UTK_gender_best_model.pt")
    
    scheduler.step()
    if current_lr != get_lr(optimizer):
        print('Loading best model weights!')
        model.load_state_dict(best_model_wts)
        
    print(f'Epoch {epoch+1}, train loss: {running_loss:.4f}, val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}, train_accuracy: {running_accuracy:.4f}')
    saving_log.append({"epoch":epoch+1, "train_loss":running_loss, "val_loss":val_loss, "val_accuracy":val_accuracy, "train_accuracy":running_accuracy})
    
#이미지 시각화 (테스트용)    
example_images=[]
gender_label={0:'male',1:'female'}
model.load_state_dict(best_model_wts)
model.eval()
a=['122.png','bb.png','angry_woman.png', 'h.png', 's2.png', 'saa.png', 'sad gir.png', 'sur.png']
for image in a:
    with torch.no_grad():

        img=cv2.imread(image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img=cv2.resize(img,(128,128))
        img=np.transpose(img,(2,0,1))
        img=torch.FloatTensor(img).to(device)
        img=torch.unsqueeze(img,0)/255.0
        
        start=time.time()
        gender_output=model(img)
        infer_time=time.time()-start

        gender_pred=gender_output.argmax(1,keepdim=True)
        gender=gender_label[gender_pred.item()]
    example_images.append(wandb.Image(
                    img, caption=f'Pred:{gender}, inference_time:{infer_time:.4f}'))
wandb.log({"Image": example_images})
