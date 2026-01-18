import os
os.environ["PYTHONUTF8"] = "1"

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
import json
from decoder_only_gpt import My_GPT_model_SFT
import torch.nn.functional as F
from tqdm import tqdm
import wandb, os, time, sys, logging
from torch.amp import autocast, GradScaler
import io


import torch._dynamo
torch._dynamo.config.suppress_errors = True

# CONFIG = {
#     "vocab_size": 32768,
#     "d_model": 512,
#     "n_layer": 12,
#     "n_head": 8,
#     "d_ff": 2048,
#     "seq_len": 512,
# }

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

device = "cuda"


CONFIG = {
    # --------------------
    # Model architecture
    # --------------------
    "model": {
        "vocab_size": 32768,
        "d_model": 512,
        "n_layer": 12,
        "n_head": 8,
        "d_ff": 2048,
        "seq_len": 512,
        "dropout": 0.1,
        "weight_tying": True,
        "norm_type": "rmsnorm",
        "ffn_type": "swiglu"
    },

    # --------------------
    # Training (SFT)
    # --------------------
    "train": {
        "batch_size": 1,
        "micro_batch_size": 1,     # future gradient accumulation
        "grad_accum_steps": 4,
        "epochs": 3,
        "lr": 2e-6,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
        "ignore_index": -100,
        "fp16": True,              # future ready
        "bf16": False,
        "seed": 42
    },

    # --------------------
    # Data
    # --------------------
    "data": {
        "dataset": "hindi_sft_v1",
        "format": "### प्रश्न: / ### उत्तर:",
        "pad_token_id": 0,
        "eos_token_id": 2,
        "max_seq_len": 512,
        "mask_prompt": True        # loss only on answer
    },

    # --------------------
    # Logging / Checkpoint
    # --------------------
    "logging": {
        "project": "HindiGPT-SFT",
        "log_every": 50,
        "eval_every": 500,
        "save_every": 1000,
        "save_dir": "checkpoints_sft",
        "wandb": True
    }
}



data = []
with open("alpaca_hindi_sft_clean.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


val_data = []
with open("hindi_sft_val.jsonl", "r", encoding="utf-8") as f1:
    for line1 in f1:
        val_data.append(json.loads(line1))
# for i in range(5):
#     print("---- SAMPLE ----")
#     print(data[i]["text"])

tokenizer = spm.SentencePieceProcessor(model_file="hindi_tokenizer_new.model")

# print(sp.encode("### प्रश्न:\n"))
# print(sp.encode("### उत्तर:\n"))

# print(tokenizer.id_to_piece(0))


ANSWER_PREFIX = "### उत्तर:"



class SFT_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]["text"]

        # 1️⃣ find answer start in STRING
        answer_key = "### उत्तर:"
        start_char = text.find(answer_key)

        if start_char == -1:
            print("⚠️ Answer prefix not found (string level)")
            print(text[:200])
            start_char = len(text)

        # move pointer AFTER "### उत्तर:"
        # answer_text = text[start_char + len(answer_key):]

        full_ids = self.tokenizer.encode(text)
        prompt_ids = self.tokenizer.encode(text[:start_char + len(answer_key)])
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_ids):] = input_ids[len(prompt_ids):]
        
        return {
            "input_ids" : input_ids,
            "labels" : labels
        }


def collate_fn(batch):
    seq_len = CONFIG["model"]["seq_len"]

    input_ids, labels = [], []

    for item in batch:
        ids = item["input_ids"][:seq_len]
        lbls = item["labels"][:seq_len]

        pad_len = seq_len - len(ids)

        if pad_len > 0:
            ids = torch.cat([
                ids,
                torch.zeros(pad_len, dtype=torch.long)
            ])

            lbls = torch.cat([
                lbls,
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

        input_ids.append(ids)
        labels.append(lbls)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels)
    }




ds1 = SFT_Dataset(data, tokenizer)
ds2 = SFT_Dataset(val_data, tokenizer)

train_loader = DataLoader(ds1, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ds2, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = My_GPT_model_SFT(
    vocab_size=CONFIG["model"]["vocab_size"], num_layers=CONFIG["model"]["n_layer"],
    d_model=CONFIG["model"]["d_model"], d_ff=CONFIG["model"]["d_ff"],
    num_heads=CONFIG["model"]["n_head"], seq_len=CONFIG["model"]["seq_len"]
)

ckpt = torch.load("checkpoints_HindiGPT-v1_step280000.pt", map_location="cpu")

state_dict = ckpt["model"]

# model.load_state_dict(state)

model.cuda()

model.train()

# 🔥 strip _orig_mod. prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)



optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01)

run_name = (
    f"HindiGPT-v1_"
    f"bs{CONFIG["train"]['batch_size']}_"
    f"lr{CONFIG["train"]['lr']}_"
    f"{time.strftime('%H%M%S')}"
)

wandb.init(
    project=CONFIG["logging"]["project"],
    name=run_name,
    config=CONFIG
)

CKPT_DIR = CONFIG["logging"]["save_dir"]
os.makedirs(CKPT_DIR, exist_ok=True)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")

    # strip _orig_mod. if present
    new_state = {}
    for k, v in ckpt["model"].items():
        if k.startswith("_orig_mod."):
            new_state[k.replace("_orig_mod.", "")] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    print(f"✅ Resumed from epoch {start_epoch}, step {global_step}")
    return start_epoch, global_step


resume_ckpt = None  # or "checkpoints_sft/last.pt"

start_epoch = 0
global_step = 0

if resume_ckpt is not None:
    start_epoch, global_step = load_checkpoint(
        resume_ckpt, model, optimizer
    )


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    losses = []

    for batch in val_loader:
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        logits = model(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, epoch, global_step, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": CONFIG
        },
        path
    )


def main():

    best_eval_loss = float("inf")

    model.train()
    global_step = 0

    for epoch in range(start_epoch, CONFIG["train"]["epochs"]):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):

            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(input_ids)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            loss = loss / CONFIG["train"]["grad_accum_steps"]

            scaler = GradScaler()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["train"]["grad_clip"])
            if (step + 1) % CONFIG["train"]["grad_accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # ---- tqdm display ----
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # ---- wandb log ----
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "epoch": epoch,
                },
                step=global_step
            )

            # ---- periodic save ----
            if global_step % CONFIG["logging"]["save_every"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    f"{CKPT_DIR}/step_{global_step}.pt"
                )

            
            # ---- evaluation ----
            if global_step % CONFIG["logging"]["eval_every"] == 0:
                eval_loss = evaluate(model, val_loader)

                wandb.log(
                    {"eval/loss": eval_loss},
                    step=global_step
                )

                print(f"\nEval loss: {eval_loss:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        f"{CKPT_DIR}/best.pt"
                    )

if __name__ == "__main__":
    main()

# for epoch in range(10):
#     for step, batch in enumerate(loader):
#         input_ids = batch["input_ids"].cuda()   # (B, S)
#         labels = batch["labels"].cuda()         # (B, S)

#         # Forward → logits
#         logits = model(input_ids)               # (B, S, V)

#         # Causal LM loss (ignore padding + question)
#         loss = F.cross_entropy(
#             logits.view(-1, logits.size(-1)),   # (B*S, V)
#             labels.view(-1),                    # (B*S)
#             ignore_index=-100
#         )

#         optimizer.zero_grad()
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         if step % 50 == 0:
#             print(f"epoch {epoch} step {step} loss {loss.item():.4f}")