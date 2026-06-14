import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.join("vendor", "interchange"))

from datasets.mqnli import MQNLIBertData, bert_subphrase_collate
from modeling.arithmetic_bert import ArithmeticBertModule

DATA_DIR = "data/mqnli"
REMAPPING_FILE = os.path.join("vendor", "interchange", "data", "tokenization", "bert-remapping.txt")
CHECKPOINT_DIR = "checkpoints"
FINETUNED_PATH = os.path.join(CHECKPOINT_DIR, "bert_mqnli_finetuned.pt")

BATCH_SIZE = 4
LR = 2e-5
EPOCHS = 3
LR_WARMUP_RATIO = 0.5


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    loss_fxn = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[4].to(device) if isinstance(batch[4], torch.Tensor) else torch.tensor(batch[4]).to(device)
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            total_loss += loss_fxn(logits[:, :3], labels).item()
            correct += (logits[:, :3].argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    train_file = os.path.join(DATA_DIR, "0gendata.train.subphrase")
    dev_file   = os.path.join(DATA_DIR, "0gendata.val")
    test_file  = os.path.join(DATA_DIR, "0gendata.test")

    for f in (train_file, dev_file, test_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found — run `python main.py generate mqnli` first (train needs .subphrase file)")

    data = MQNLIBertData(train_file, dev_file, test_file, REMAPPING_FILE, variant="subphrase")
    print(f"train: {len(data.train)}  dev: {len(data.dev)}  test: {len(data.test)}")

    model = ArithmeticBertModule(num_labels=10).to(device)
    model.bert.resize_token_embeddings(len(data.tokenizer))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader = DataLoader(data.train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=bert_subphrase_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fxn = nn.CrossEntropyLoss(reduction='none')

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(LR_WARMUP_RATIO * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_dev_acc = 0.0

    #TEMP
    # torch.save(model.state_dict(), "checkpoints/bert_mqnli_pretrained.pt")
    # print("SAVED")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}", leave=False)

        for step, batch in enumerate(bar, 1):
            input_ids      = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            loss_weighting = batch[3].to(device)
            labels         = batch[5].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            loss = (loss_fxn(logits, labels) * loss_weighting).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total += labels.size(0)

            if step % 5 == 0:
                bar.set_postfix(loss=f"{total_loss / step:.4f}")

        train_loss = total_loss / len(train_loader)
        dev_acc, dev_loss = evaluate(model, data.dev, device)
        print(f"epoch {epoch + 1}/{EPOCHS} | train loss {train_loss:.4f} | val loss {dev_loss:.4f} | val acc {dev_acc:.3f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), FINETUNED_PATH)
            print(f"  ↑ best model saved (val acc {best_dev_acc:.3f})")

    print(f"saved best model → {FINETUNED_PATH}")

    model.load_state_dict(torch.load(FINETUNED_PATH))
    test_acc, test_loss = evaluate(model, data.test, device)
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")
