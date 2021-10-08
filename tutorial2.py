import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.onnx as onnx
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()


# epoch is number of iterations, batch_size is size of batch, learning_rate is speed of learning. Simple eh?
# Loss is how far off guess is from real answer. Try to minimize this. Many loss functions out there.
# Optimizer adjusts parameters to reduce model error at each step of training. Different algortihms for this.
learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# optimizer.zero_grad() resets gradient, Don't want them to add up every time.
# loss.backward() gets your gradient
# optimizer.step() adjusts your params based on that gradient


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


# Saving and loading the model

# Models save learned params in state_dict. Can get them through torch.save

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# To load these model weights do this

# We do not specify pretrained=True, i.e. do not load default weights
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# To save the sturcture of class with model do this. Might use python pickle module. Funny name
torch.save(model, 'model.pth')

# To load it do this
model = torch.load('model.pth')

# To export a model through ONNX you do this. You gotta have a test variable of right size tho
input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')


"""
More info:

nn.Flatten: Converts image like 28x28 to 784 pixels
nn.linear: Applies linear transformation. Basically turns it into a tensor for your neural network to work on. Mulitplies through some wack equations.
nn.ReLU: Apples after linear transformation to introduce nonlinearity, so neural network gets to experience variety
nn.Sequential: Allows you to do all of above in one function
nn.Softmax: Scales logits into values [0,1]
dimL parameter indixates dimension which values must sum to 1


Cool stuff:

self.hidden = nn.Linear(784, 256): This creates hidden layer in your neural network. Helpful for adversarial ML.
nn.Conv2d = Applies a 2D convolution over an input signal composed of several input planes.
"""
