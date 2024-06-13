import sys
from PPairS.datasets import IPCCDataset
from PPairS.trainer import Trainer
from PPairS.classify import CompareClassifier

import torch as t
import torch.nn.functional as F


n_epoch = int(sys.argv[1])
device = sys.argv[2]
trainer = Trainer(
    dataset_class = IPCCDataset,
    dataset_kwargs = {"subset": ["AR6"], "contrast": False, "device": device},
    splits = (1., 1.),
    batch_size = 64
)
clf = CompareClassifier(device=device)
opt = t.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.001)
for epoch in range(n_epoch):
    for batch, labels in trainer.train_loader:
        logits = clf(batch)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), (labels > 0.5).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    accuracy = []
    for batch, labels in trainer.train_loader:
        with t.inference_mode(): logits = clf(batch)
        P = F.sigmoid(logits)
        accuracy.append(((P.squeeze(-1) > 0.5) == (labels > 0.5)).float().mean())
    accuracy = sum(accuracy) / len(accuracy)
    print(f"epoch {epoch+2} ({round(accuracy.item(), 3)})")
t.save(clf.state_dict(), f"/gws/nopw/j04/ai4er/users/maiush/PPairS/src/dev/ipcc_binary_search/clf_{n_epoch}.pt")