# lr_finder.py
import torch
import torch.optim as optim
from torch_lr_finder import LRFinder

# Simple model and data
data = [(torch.randn(10), torch.tensor(1)) for _ in range(100)]
loader = torch.utils.data.DataLoader(data, batch_size=10)

model = torch.nn.Linear(10, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7)

# LR Finder
lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(loader, end_lr=1, num_iter=100)
lr_finder.plot()

# one_cycle_lr.py
from torch.optim.lr_scheduler import OneCycleLR

# One-Cycle Learning Rate
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(loader), epochs=10)
for epoch in range(10):
    for batch in loader:
        optimizer.step()
        scheduler.step()
