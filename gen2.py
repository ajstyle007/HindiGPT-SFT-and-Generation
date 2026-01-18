import torch
import torch.nn.functional as F
import sentencepiece as spm
from decoder_only_gpt import My_GPT_model_SFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_PATH = "checkpoints_HindiGPT-v1_step295000.pt"
CKPT_PATH = "checkpoints_HindiGPT-v1_step280000.pt"
SEQ_LEN = 512

TEMPERATURE = 0.6
TOP_P = 0.85
REPETITION_PENALTY = 1.6

# Sampling hyperparameters
  # ← Increase this! 1.2–1.8 works best for repetitive small models
PENALTY_WINDOW = 128       # Not used directly now, but kept for future
MAX_NEW_TOKENS = 200

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp = spm.SentencePieceProcessor(model_file="hindi_tokenizer_new.model")

print("BOS:", sp.bos_id(), "EOS:", sp.eos_id(), "PAD:", sp.pad_id())



@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()
    eos_id = sp.eos_id()
    prompt_len = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -SEQ_LEN:]
        logits = model(idx_cond)[:, -1, :] / TEMPERATURE

        logits[logits < -20] = -1e9
        # block non-Devanagari tokens
        for tid in range(sp.get_piece_size()):
            piece = sp.id_to_piece(tid)
            if not any('\u0900' <= ch <= '\u097F' for ch in piece):
                logits[0, tid] -= 5.0

        # repetition penalty ONLY on generated tokens
        generated_tokens = idx[0, prompt_len:].tolist()
        for t in set(generated_tokens):
            logits[0, t] /= REPETITION_PENALTY

        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cum_probs > TOP_P
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(sorted_probs, 1)
        next_id = torch.gather(sorted_indices, -1, next_id)

        idx = torch.cat([idx, next_id], dim=1)

        if next_id.item() == eos_id:
            break

    return idx

    
def main():
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    print("Initializing model...")
    model = My_GPT_model_SFT(
        vocab_size=sp.get_piece_size(),
        num_layers=12,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        seq_len=SEQ_LEN
    ).to(DEVICE)

    # Handle torch.compile prefix
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("Detected torch.compile prefix, stripping '_orig_mod.'...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

    # prompt = "नमस्ते"
    # prompt = """प्रश्न: भारत की राजधानी क्या है?
    #         उत्तर:"""
    prompt = "### प्रश्न:\nस्वस्थ रहने के लिए तीन सुझाव दें।\n\n### उत्तर:\n"

    print(f"Prompt: {prompt}")

    input_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    x = torch.tensor([input_ids], device=DEVICE)
    prompt_len = x.shape[1]

    for i in range(3):
        print(f"--- Generation {i+1}/3 ---")
        output_ids = generate(model, x.clone(), MAX_NEW_TOKENS)
        generated_ids = output_ids[0, prompt_len:].tolist()
        generated_text = sp.decode(generated_ids)
        print(generated_text)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()





