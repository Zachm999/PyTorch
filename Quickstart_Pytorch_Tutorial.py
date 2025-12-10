import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#Downloads training data from PyTorch open datasets. FashionMNIST is class (basically a blueprint for a type of object). So in this step we are creting an object using the FashionMNIST class stored in the datasets module and storing it to the training_data variable. 
training_data = datasets.FashionMNIST(
    root="data" ,
    train=True ,
    download=True ,
    transform=ToTensor() ,
)

#Download Test data from Pytorch open dataset. 
test_data = datasets.FashionMNIST(
    root='data' ,
    train=False ,
    download=True ,
    transform=ToTensor()
)


#After establishing the datasets (training and test), our next job is to pass the datasets to the dataloader. This wraps an interable over the object allowing us to use normal python tools like for loops to manipulate the data. 

#batch size is the number of examples that will be propogated (fed to the network) as it is being trained. A smaller batch size requires less memory but requires more time. A larger batch size requires more memory but less time per round of training. 
batch_size = 64

#create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#After loading the data and wrapping it in an interable (making it changeable using common python tools), we will not actually intiallize the model! 

#First we check what type of device we are using for training. In this case it is MPS (Metal Performance Shaders). This is Apples backend to enable GPU acceleration of training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"using {device} device")


#Define model 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return(logits)
    
model = NeuralNetwork().to(device)
print(model)

#to train the model we need to define a loss function and an optimizer 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#This step checks the model's performance against the test dataset 

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: \n {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#we want to see the accuracy increase and the loss decrease after every epoch 

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1} \n------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done!")


#Saving the model 
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#Loading the model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))