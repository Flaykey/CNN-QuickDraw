import torch
from torch import nn
import numpy as np
from torch import optim
import draw

device = "cuda" if torch.cuda.is_available() else "cpu"
class CNN(nn.Module):
    
    def __init__(self,number_of_claseses,lr):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1,8,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8,16,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(16*7*7,64),
                nn.ReLU(),
                nn.Linear(64,number_of_claseses)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr)
    def forward(self,x):
        return self.model(x)

        


        
#data preparation
path = "dataset/full_numpy_bitmap_"
categories = ["airplane","apple","bicycle","broom","banana","bed","basket","baseball","axe","angel","alarm clock","ambulance"]
traindatasetX = []
traindatasetY = []

testdatasetX = []
testdatasetY = []


for i in range(0,len(categories)):
    x = np.load(path + categories[i] + ".npy")[:20000]
    print(x.shape)
    l = len(x)
    trainsize = int(l * 0.6)
    trainX = x[:trainsize]
    testX = x[trainsize:]
    traindatasetX.append(trainX)
    traindatasetY.append(np.full(trainsize,i))
    testdatasetX.append(testX)
    testdatasetY.append(np.full(l-trainsize,i))

traindatasetX = np.concatenate(traindatasetX, axis=0)
traindatasetY = np.concatenate(traindatasetY, axis=0)
testdatasetX = np.concatenate(testdatasetX, axis=0)
testdatasetY = np.concatenate(testdatasetY, axis=0)

perm = np.random.permutation(len(traindatasetX))

traindatasetX = traindatasetX[perm]
traindatasetY = traindatasetY[perm]


traindatasetX = torch.tensor(traindatasetX, dtype=torch.float32).reshape(-1,1,28,28) / 255.0
traindatasetY = torch.tensor(traindatasetY, dtype=torch.long)
testdatasetX = torch.tensor(testdatasetX, dtype=torch.float32).reshape(-1,1,28,28) /255.0
testdatasetY = torch.tensor(testdatasetY, dtype=torch.long)

traindatasetX = traindatasetX
traindatasetY = traindatasetY

testdatasetX = testdatasetX.to(device)
testdatasetY = testdatasetY.to(device)



epochs = 1
lr = 0.001
model = CNN(len(categories),lr).to(device)

batch_size = 32
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    count = 0
    for i in range(0,len(traindatasetX),batch_size):
        model.optimizer.zero_grad()

        x = traindatasetX[i:i+batch_size].to(device)
        y = traindatasetY[i:i+batch_size].to(device)

        output = model(x)
        loss = model.loss_fn(output,y)
        epoch_loss += loss.item()
        loss.backward()
        model.optimizer.step()
        count += 1
        del x, y, output, loss
        torch.cuda.empty_cache()
    epoch_loss/=count
    print(f"Epoch {epoch + 1} loss : {epoch_loss}")

model.eval()
with torch.no_grad():
    pred = model(testdatasetX)
    preds = torch.argmax(pred, dim=1)
    acc = (preds == testdatasetY).float().mean()
    print("Test accuracy:", acc.item())

model.to("cpu")

window = draw.DrawScreen(700,700,28)
window.run(model=model,categories=categories)