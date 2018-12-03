import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
from torchvision import datasets, transforms, models


test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


test_data = datasets.ImageFolder('Evaluate', transform=test_transforms)

testloader = torch.utils.data.DataLoader(test_data, batch_size=1)

model = models.vgg11(pretrained=True)


print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 500),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(500,2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),

                                 nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

print(model)

model.load_state_dict(torch.load('pedestrianModel.pth'))
model.to(device);

epochs = 1
steps = 0
running_loss = 0
print_every = 5
test_loss = 0
accuracy = 0

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)

        print(top_p,top_class)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

