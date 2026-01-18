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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import io, math
import torch._dynamo
torch._dynamo.config.suppress_errors = True


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        "grad_accum_steps": 1,
        "epochs": 4,
        "lr": 5e-5,
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


tokenizer = spm.SentencePieceProcessor(model_file="hindi_tokenizer_new.model")


class SFT_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        
        self.data = data
        self.tokenizer = tokenizer
        self.answer_key = "### उत्तर:"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]["text"]

        full_ids = self.tokenizer.encode(text)

        # Find where answer starts in **string**
        
        pos = text.find(self.answer_key)
        if pos == -1:
            pos = len(text)
            print("Warning: No answer prefix found!")
            

        prompt_text = text[:pos + len(self.answer_key)]
        prompt_ids = self.tokenizer.encode(prompt_text)

        min_len = min(len(prompt_ids), len(full_ids))

        answer_start = 0
        for i in range(min_len, 0, -1):
            if prompt_ids[:i] == full_ids[:i]:
                answer_start = i
                break

        # Safety fallback: if no match, don't mask (train on everything)
        if answer_start == 0:
            print(f"Warning: No prefix match for sample {index}! Training on full sequence.")
            labels = torch.tensor(full_ids, dtype=torch.long)
        else:
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            labels[answer_start:] = input_ids[answer_start:]

        # Truncate/pad as before...
        # (your existing padding code here)

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    seq_len = CONFIG["model"]["seq_len"]
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    pad_id = CONFIG["data"]["pad_token_id"]   # ← yahan se le

    for item in batch:
        ids = item["input_ids"][:seq_len]
        lbls = item["labels"][:seq_len]

        curr_len = len(ids)
        pad_len = seq_len - curr_len

        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)])

        # Attention mask: 1 = real token, 0 = padding
        mask = torch.ones(seq_len, dtype=torch.long)
        if pad_len > 0:
            mask[curr_len:] = 0

        input_ids_list.append(ids)
        labels_list.append(lbls)
        attention_mask_list.append(mask)

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list)
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

model.to('cuda')


pretrained_path = "checkpoints_HindiGPT-v1_step280000.pt"  # <-- put actual pretrained checkpoint path here

if os.path.isfile(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # If checkpoint has key "model" holding weights, else use checkpoint directly
    state_dict = checkpoint.get("model", checkpoint)
    
    # Remove any unwanted prefixes (like _orig_mod.) if present, matching your load_checkpoint code
    new_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"Loaded pretrained model with missing keys: {missing}, unexpected keys: {unexpected}")
else:
    print("Pretrained model checkpoint not found, training from scratch.")

    

scaler = GradScaler(enabled=CONFIG["train"]["fp16"])

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["train"]["lr"], weight_decay=0.01)

warmup_steps = 300
total_steps = len(train_loader) * CONFIG["train"]["epochs"] // CONFIG["train"]["grad_accum_steps"]

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=warmup_steps
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-8
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-8)

run_name = f"HindiGPT-SFT-v1_bs{CONFIG['train']['batch_size']}_lr{CONFIG['train']['lr']}_{time.strftime('%H%M%S')}"
wandb.init(project=CONFIG["logging"]["project"], name=run_name, config=CONFIG)

CKPT_DIR = CONFIG["logging"]["save_dir"]
os.makedirs(CKPT_DIR, exist_ok=True)





# ─── Checkpoint functions ──────────────────────────────────
def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Loaded epoch: {ckpt.get('epoch')}, step: {ckpt.get('global_step')}")

    # Handle _orig_mod. prefix (torch.compile case)
    new_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(new_state)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("global_step", 0)
    print(f"✓ Resumed from epoch {epoch}, step {step}")
    return epoch, step

def save_checkpoint(model, optimizer, scaler, epoch, global_step, path, is_best=False):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),          # ← bahut zaroori
        "epoch": epoch,
        "global_step": global_step,
        "config": CONFIG,
        "is_best": is_best,
    }, path)
    print(f"→ Saved checkpoint @ step {global_step} {'(best)' if is_best else ''}")

    wandb.save(path)  # This uploads the checkpoint file as an artifact to your wandb run


# ─── Evaluation ────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        attn_mask = batch["attention_mask"].cuda()   

        # real_tokens = (labels != -100).sum().item()
        # total_tokens = labels.numel()

        # print(f"Eval batch | Real target tokens: {real_tokens}/{total_tokens} "
        #       f"({real_tokens/total_tokens*100:.1f}%)")
        # if real_tokens == 0:
        #     print("Warning: No real target tokens in this eval batch! Loss will be 0.")


        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attn_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        total_loss += loss.item()
        count += 1

    model.train()
    avg_loss = total_loss / count if count > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')
    return avg_loss, perplexity


# ─── Main training ─────────────────────────────────────────
def main():
    start_epoch = 0
    global_step = 0
    best_eval_loss = float("inf")

    resume_ckpt = "checkpoints_sft/last.pt"       # ← yahan apna last checkpoint daal dena

    if resume_ckpt  and os.path.exists(resume_ckpt):
        start_epoch, global_step = load_checkpoint(
            resume_ckpt, model, optimizer, scaler
        )

    model.train()

    steps_per_epoch = len(train_loader) // CONFIG["train"]["grad_accum_steps"]
    total_steps = steps_per_epoch * CONFIG["train"]["epochs"]

    for epoch in range(start_epoch, CONFIG["train"]["epochs"]):
        accum_loss = 0.0
        accum_count = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['train']['epochs']} "
                 f"(global step {global_step:,})",
            initial=global_step % steps_per_epoch,
            total=len(train_loader)
            )

        for batch_idx, batch in enumerate(progress_bar):
            global_step += 1

            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attn_mask = batch["attention_mask"].to(device)

            # # Add this:
            # real_tokens = (labels != -100).sum().item()
            # total_tokens = labels.numel()
            # print(f"Step {global_step} | Real target tokens: {real_tokens}/{total_tokens} "
            #     f"({real_tokens/total_tokens*100:.1f}%)")
            
            # if real_tokens == 0:
            #     print("Warning: No real target tokens in this batch! Skipping loss computation.")
            #     continue  # Skip this batch if no real targets to avoid fake 0 loss

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attn_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.1
                )
            
            # === IMPORTANT DIAGNOSTICS (add these!) ===
            if global_step % 20 == 0:  # print every 20 steps to not spam
                real_mask = labels.view(-1) != -100
                if real_mask.any():
                    real_logits = logits.view(-1, logits.size(-1))[real_mask]
                    max_logit = real_logits.max(dim=-1).values.mean().item()
                    top1_acc = (real_logits.argmax(dim=-1) == labels.view(-1)[real_mask]).float().mean().item() * 100
                    train_ppl = math.exp(loss.item())
                    print(f"Step {global_step:5d} | "
                          f"Loss: {loss.item():.6f} | "
                          f"PPL: {train_ppl:.4f} | "
                          f"Max logit: {max_logit:.3f} | "
                          f"Top-1 acc: {top1_acc:.2f}%")

            loss = loss / CONFIG["train"]["grad_accum_steps"]
            scaler.scale(loss).backward()

            accum_loss += loss.item()
            accum_count += 1

            # Gradient clipping & step
            if (batch_idx + 1) % CONFIG["train"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["train"]["grad_clip"])

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Real average loss for logging
                avg_loss = accum_loss / accum_count
                avg_ppl = math.exp(avg_loss)
                accum_loss = 0.0
                accum_count = 0

                wandb.log({"train/loss": avg_loss, "train/perplexity": avg_ppl}, step=global_step)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{avg_ppl:.4f}")

            # Periodic save
            if global_step % CONFIG["logging"]["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, scaler, epoch, global_step,
                    f"{CKPT_DIR}/step_{global_step}.pt"
                )
                # Optional: last.pt update
                save_checkpoint(
                    model, optimizer, scaler, epoch, global_step,
                    f"{CKPT_DIR}/last.pt"
                )

            # Evaluation
            if global_step % CONFIG["logging"]["eval_every"] == 0:
                eval_loss, eval_ppl = evaluate(model, val_loader)
        
                wandb.log({"eval/loss": eval_loss, "eval/perplexity": eval_ppl}, step=global_step)
                print(f"Eval loss @ step {global_step}: {eval_loss:.4f} | PPL: {eval_ppl:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        model, optimizer, scaler, epoch, global_step,
                        f"{CKPT_DIR}/best.pt", is_best=True
                    )

if __name__ == "__main__":
    main()