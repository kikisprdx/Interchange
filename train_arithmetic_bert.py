import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from reproduction_datasets.arithmetic import ArithmeticData, generate_data
from modeling.arithmetic_bert import ArithmeticBertModule

# paths
DATA_FILE = "data/arithmetic.csv"
VOCAB_DIR = "data/arithmetic_vocab"
CHECKPOINT_DIR = "checkpoints"
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "bert_arithmetic_pretrained.pt")
FINETUNED_PATH = os.path.join(CHECKPOINT_DIR, "bert_arithmetic_finetuned.pt")

# hyperparameters
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3
LR_WARMUP_RATIO = 0.5


def evaluate(model, dataset, device, loss_fxn):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    correct = total = 0
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[4].to(device)
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            total_loss += loss_fxn(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / len(loader)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    if not os.path.exists(DATA_FILE):
        os.makedirs("data", exist_ok=True)
        generate_data()
        print(f"Generated {DATA_FILE}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens_to_add = [str(i) for i in range(-200, 201) if str(i) not in tokenizer.vocab]
    tokenizer.add_tokens(tokens_to_add)
    os.makedirs(VOCAB_DIR, exist_ok=True)
    vocab_path = tokenizer.save_vocabulary(VOCAB_DIR)[0]
    # save_vocabulary only writes self.vocab, not added_tokens — append manually
    # so ArithmeticData's BertTokenizer(vocab_path) finds these tokens
    with open(vocab_path, "a", encoding="utf-8") as f:
        for token in tokens_to_add:
            f.write(token + "\n")

    data = ArithmeticData(DATA_FILE, vocab_path)
    print(f"train: {len(data.train)}  dev: {len(data.dev)}  test: {len(data.test)}")

    model = ArithmeticBertModule(num_labels=3).to(device)

    cfg = model.config()
    print(
        f"\nhyperparams: epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}  "
        f"labels={cfg['num_labels']}  dropout={cfg['dropout']}  "
        f"hidden={cfg['hidden_size']}  bert={cfg['bert_model']}\n"
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), PRETRAINED_PATH)
    print(f"Saved pretrained weights → {PRETRAINED_PATH}")

    train_loader = DataLoader(data.train, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fxn = nn.CrossEntropyLoss()

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(LR_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}", leave=False)

        for step, batch in enumerate(bar, 1):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[4].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fxn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total += labels.size(0)

            if step % 5 == 0:
                global_step = (epoch * len(train_loader)) + step
                bar.set_description(
                    f"epoch {epoch + 1}/{EPOCHS} ({global_step}/{total_steps} steps)"
                )

        train_loss = total_loss / len(train_loader)
        dev_acc, dev_loss = evaluate(model, data.dev, device, loss_fxn)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train loss {train_loss:.4f} "
            f"| val loss {dev_loss:.4f} "
            f"| val acc {dev_acc:.3f}"
        )

    torch.save(model.state_dict(), FINETUNED_PATH)
    print(f"Saved finetuned weights → {FINETUNED_PATH}")

    test_acc, test_loss = evaluate(model, data.test, device, loss_fxn)
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
