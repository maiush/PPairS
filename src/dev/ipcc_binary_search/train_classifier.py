import sys
from PPairS.datasets import IPCCDataset
from PPairS.trainer import Trainer
from PPairS.classify import CompareClassifier

import torch as t
import torch.nn.functional as F

from tqdm import tqdm


trainer = Trainer(
    dataset_class = IPCCDataset,
    dataset_kwargs = {"subset": ["AR6"], "contrast": False, "device": "cpu"},
    splits = (1., 1.),
    batch_size = 64
)

n_epoch = int(sys.argv[1])
device = sys.argv[2]
clf = CompareClassifier(device=device)
opt = t.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.001)
desc = "epoch 1"
for epoch in range(n_epoch):
    bar = tqdm(trainer.train_loader)
    bar.set_description(desc)
    for batch, labels in bar:
        logits = clf(batch)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), (labels > 0.5).float())
        opt.zero_grad()
        loss.backward()
        opt.step()

    # validation accuracy
    accuracy = []
    for batch, labels in trainer.val_loader:
        with t.no_grad(): logits = clf(batch)
        P = F.sigmoid(logits)
        accuracy.append(((P.squeeze(-1) > 0.5) == (labels > 0.5)).float().mean())
    accuracy = sum(accuracy) / len(accuracy)
    desc = f"epoch {epoch+2} ({round(accuracy.item(), 3)})"

t.save(clf.state_dict(), f"/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/ipcc_binary_search/clf_{n_epoch}.pt")