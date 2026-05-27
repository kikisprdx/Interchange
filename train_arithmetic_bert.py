import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets.arithmetic import ArithmeticData, generate_data
from modeling.arithmetic_bert import ArithmeticBertModule

# paths
DATA_FILE = "data/arithmetic.csv"
VOCAB_DIR = "/tmp/arithmetic_vocab"
CHECKPOINT_DIR = "checkpoints"
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "bert_arithmetic_pretrained.pt")
FINETUNED_PATH = os.path.join(CHECKPOINT_DIR, "bert_arithmetic_finetuned.pt")

# hyperparameters
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)
            logits, _  = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_FILE):
        os.makedirs("data", exist_ok=True)
        generate_data()
        print(f"Generated {DATA_FILE}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    os.makedirs(VOCAB_DIR, exist_ok=True)
    vocab_path = tokenizer.save_vocabulary(VOCAB_DIR)[0]

    data = ArithmeticData(DATA_FILE, vocab_path)
    print(f"train: {len(data.train)}  dev: {len(data.dev)}  test: {len(data.test)}")

    model = ArithmeticBertModule(num_labels=3).to(device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), PRETRAINED_PATH)
    print(f"Saved pretrained weights → {PRETRAINED_PATH}")

    train_loader = DataLoader(data.train, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fxn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0

        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)  # ignore hidden_states
            loss = loss_fxn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        dev_acc = evaluate(model, data.dev, device)

        print(f"Epoch {epoch + 1}/{EPOCHS} "
              f"| train loss {train_loss:.4f} "
              f"| train acc {train_acc:.3f} "
              f"| dev acc {dev_acc:.3f}")

    # save finetuned weights
    torch.save(model.state_dict(), FINETUNED_PATH)
    print(f"Saved finetuned weights → {FINETUNED_PATH}")

    # final test accuracy
    test_acc = evaluate(model, data.test, device)
    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
