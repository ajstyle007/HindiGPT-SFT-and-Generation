import re
import fitz
import numpy as np
import torch
import faiss
import sentencepiece as spm
from sentence_transformers import SentenceTransformer, CrossEncoder

import torch.nn.functional as F
from decoder_only_gpt import My_GPT_model
from gen_func import build_index, embed_model, fix_pdf_text, sentence_chunk_text, clean_hindi_text,hard_clean

sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = My_GPT_model(
    vocab_size=sp.get_piece_size(),
    num_layers=12,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    seq_len=512
).to(DEVICE)

# Load final SFT checkpoint
model.load_state_dict(torch.load("full_sft_final.pt", map_location=DEVICE))
model.eval()



@torch.inference_mode()
def generate(
    model,
    input_ids,                  # shape: [batch_size, seq_len]
    max_new_tokens=120,
    temperature=0.75,           # 0.2 बहुत कम → repetitive → 0.7–1.0 ज्यादातर बेहतर
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.15,    # 1.1–1.2 के बीच ज्यादातर अच्छा काम करता है
    eos_token_id=None,          # स्पष्ट रूप से पास करना बेहतर
    pad_token_id=None,
    device=None
):
    if device is None:
        device = input_ids.device

    model.eval()
    generated = input_ids.clone().to(device)
    batch_size = generated.size(0)

    # अगर eos_token_id नहीं दिया तो model config से लेने की कोशिश
    # if eos_token_id is None and hasattr(model.config, "eos_token_id"):
    #     eos_token_id = model.config.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # आखिरी टोकन के logits
        next_logits = logits[:, -1, :]   # [batch, vocab]

        # repetition penalty (सही तरीका – पहले से इस्तेमाल हुए टोकन्स पर)
        if repetition_penalty != 1.0:
            for i in range(batch_size):
                seen = generated[i].unique()
                next_logits[i, seen] /= repetition_penalty   # पहले से आए टोकन को दबाना
                # कुछ लोग penalty > 1 को divide करते हैं, < 0 को multiply — लेकिन divide ज्यादातर स्टेबल

        # temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # NaN / Inf बचाव
        if torch.isnan(next_logits).any() or torch.isinf(next_logits).any():
            print("Warning: NaN/Inf detected in logits → stopping")
            break

        probs = F.softmax(next_logits, dim=-1)

        # Top-k filtering
        if top_k > 0 and top_k < probs.size(-1):
            top_k_vals, top_k_idx = torch.topk(probs, top_k, dim=-1)
            probs = torch.zeros_like(probs).scatter_(-1, top_k_idx, top_k_vals)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens after cumulative prob > top_p
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_probs = sorted_probs.masked_fill(mask, 0.0)

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Sample
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_token)
        else:
            # greedy या temperature-only
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        # EOS पर रुकना (batch में अलग-अलग हो सकता है)
        if eos_token_id is not None:
            is_done = (next_token == eos_token_id)
            if is_done.all():
                break

    return generated









def get_answer(query):
    # PDF लोड + क्लीन (हर बार query पर नहीं करना चाहिए, लेकिन अभी तुम्हारे स्टाइल में डाल रहे हैं)
    pdf_path = "story.pdf"
    doc = fitz.open(pdf_path)
    raw_text = ""
    for page in doc:
        raw_text += page.get_text("text")
        # raw_text += page.get_text("unicode")
    doc.close()
    
    fixed_text = fix_pdf_text(raw_text)
    cleaned_text = clean_hindi_text(fixed_text)
    
    # Chunking (तुम्हारा sentence level वाला इस्तेमाल कर रहे हैं)
    chunks = sentence_chunk_text(cleaned_text, max_chars=800, overlap_sentences=1)
    
    # Index बनाओ (हर बार बनाना inefficient है, लेकिन तुम्हारे मूल कोड के हिसाब से)
    global index  # अगर पहले से बना हो तो reuse, नहीं तो नया
    if 'index' not in globals() or index is None:
        index, chunks = build_index(chunks)   # chunks को update भी कर लेता है
    
    # Embed query (तुम्हारा तरीका)
    cleaned_query = query
    query_embedding = embed_model.encode([cleaned_query])
    
    # Search
    D, I = index.search(query_embedding, k=1)   # तुम्हारा मूल k=1
    
    retrieved_chunks = [chunks[i] for i in I[0]]
    
    # Print retrieved chunks (तुम्हारा मूल print)
    for idx, ch in zip(I[0], retrieved_chunks):
        print(f"\n--- Chunk {idx} ---\n{ch}\n")
    
    # context ids (तुम्हारा मूल तरीका)
    context_ids = []
    for chunk in retrieved_chunks:
        context_ids += sp.encode("[संदर्भ] " + chunk + "\n")
    
    # prompt ids (तुम्हारा मूल prompt बिल्कुल वैसा ही)
    retrieved_text = "\n".join(retrieved_chunks)

    prompt = f"""
    ### प्रश्न:
    नीचे दिए गए अनुच्छेद के आधार पर 2–3 वाक्यों में बताइए कि

    ### अनुच्छेद:
    {retrieved_text}

    ### उत्तर:
    """

    prompt = hard_clean(prompt)
    prompt_ids = sp.encode(prompt)


    print("\n=== Prompt IDs की लंबाई और कुछ शुरुआती tokens ===")
    print("Length:", len(prompt_ids))
    print("First 20 tokens:", prompt_ids[:20])
    print("Decoded without junk attempt:", sp.decode(prompt_ids[:100]).replace("⁇", ""))


    # 1. Retrieved chunks print करो (already तेरा print है, लेकिन बेहतर बना देते हैं)
    print("\n=== Retrieved Context (जो model को दिख रहा है) ===")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"Chunk {i} (index {I[0][i-1]}):")
        print(chunk.strip())
        print("-" * 80)

    # 2. Context ids से बना पूरा prompt text decode करके print करो
    full_prompt_text = sp.decode(prompt_ids)
    
    print("\n=== पूरा Prompt जो model को input जा रहा है ===")
    print(full_prompt_text)
    print("=" * 100)
    
    
    # input_ids torch tensor में
    input_ids = torch.tensor([prompt_ids]).to(DEVICE)   # DEVICE तुम्हारे कोड में define होना चाहिए (जैसे "cuda" या "cpu")
    
    # तुम्हारा generate फंक्शन कॉल (बिल्कुल वैसा ही)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=120,           # ← बहुत कम करो, repetition जल्दी शुरू होता है
        temperature=0.5,              # 0.5 से थोड़ा ऊपर — ज्यादा deterministic
        top_p=0.85,
        repetition_penalty=1.3,      # 1.1 से थोड़ा बढ़ाओ लेकिन 1.5 मत जाना
        top_k=30                      # कम tokens → focus बेहतर
    )
    
    # decode सिर्फ नए tokens
    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    answer = sp.decode(generated_ids)
    
    return answer


# इस्तेमाल का तरीका
# query = "कविता की क्या परिभाषा हो सकती है?"
# answer = get_answer(query)
# print("Generated answer:", answer)