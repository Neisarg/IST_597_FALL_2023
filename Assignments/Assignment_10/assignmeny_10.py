import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# select device as cuda
# If running on cpu change this to torch.device("cpu")
device = torch.device("cuda:0")

# set random seeds
seed = 1234 ## change this seed when you run trials
random.seed(seed)
torch.manual_seed(seed)


class LinearFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, W, b):
    """
    # x -> input matrix of size n_samples x sdim
    # W -> transformation matrix
    # b -> bias term 
    """
    ctx.save_for_backward(x, W, b)
    # Write your affine transformation here:
    #-------------------
    z = None
    #-------------------
    return z
    
  @staticmethod
  def backward(ctx, grad_output):
    x, W, b = ctx.saved_tensors
   
    # Write gradient updates here:
    #----------------------------
    grad_x = None
    grad_w = None
    grad_b = None
    #-----------------------------
    return grad_x, grad_w, grad_b


# This class uses Linear Function defined above and converts it into compatible Linear Layer 
# which can be used to create MLP. You do not need to change this class.
# Look at the initial values given to  𝑊  and  𝑏 

class CustomLinearLayer(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super(CustomLinearLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    w = torch.normal(mean = 0, std = 0.1, size = [in_features, out_features], requires_grad=True)
    b = torch.full([out_features], 0.01, requires_grad=True)
    w = torch.nn.Parameter(w)
    b = torch.nn.Parameter(b)
    self.register_parameter('w', w)
    self.register_parameter('b', b)
    self.linear_function = LinearFunction.apply
  
  def forward(self, x):
    return self.linear_function(x, self.w, self.b)
  

## This class creates an MLP. Feel free to change the number of layers and their sizes
class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    # Add Linear layers below 
    #-----------------------------------
    self.layer_1 = None
    self.layer_2 = None
    # ------------------------------------
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self, x):
    # Apply layers defined above:
    #-------------------------------
    output = None
    #-------------------------------
    return output


## Loss Function
class CrossEntropyLoss(torch.nn.Module):
  def __init__(self):
    super(CrossEntropyLoss, self).__init__()
  
  def forward(self, probs, y):
    #Write loss function here:
    #-------------------------------
    loss = None
    #-------------------------------
    return loss
  

if __name__ == "__main__":
    ## Loading Fashion MNIST Data
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

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


    # split mnist trainset into two sets: mnist trainset -> 50000 , mnist validation set -> 10000
    #----------------
    train_data, val_data = None

    #----------------

    model = NeuralNetwork().to(device)
    ce_loss = CrossEntropyLoss()

    batch_size = 1024
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    valid_loader = DataLoader(val_data, batch_size=batch_size)
    max_epochs = 10
    learning_rate = 0.01

    for epoch in range(max_epochs):
        for idx, data in enumerate(train_loader):
            features, labels = data

            features = features.cuda()

            labels = labels.cuda()
            probs = model(features.reshape([-1, 784]))

            loss = ce_loss(probs, labels)
            print("Epoch {0}/{1} Iteration {2}/{3} Loss {4}: ".format(epoch, max_epochs, idx, len(train_loader), loss))

            for param in model.parameters():
                param.grad = None

            loss.backward()

            for name, param in model.named_parameters():
            # Write paramtere update routine here:
            # --------------
                new_param = None
            
            # --------------
            with torch.no_grad():
                param.copy_(new_param)

        
        # Write Validation routine here: 
        #----------------------------


    # Write evaluation code for Test set here
    # ---------------------------------------
    # 


    # Feel free to change this routine to demonstrate underfitting and overfitting
    

