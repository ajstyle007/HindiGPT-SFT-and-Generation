import torch
import torch.nn.functional as F
import sentencepiece as spm
from decoder_only_gpt import My_GPT_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints_HindiGPT-v1_step280000.pt"
# CKPT_PATH = "checkpoints_sft/step_10000.pt"
SEQ_LEN = 512

# Sampling hyperparameters
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.2   # ← Increase this! 1.2–1.8 works best for repetitive small models
PENALTY_WINDOW = 128       # Not used directly now, but kept for future
MAX_NEW_TOKENS = 200

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()
    original_len = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -SEQ_LEN:]
        logits = model(idx_cond)[:, -1, :] / TEMPERATURE  # [1, vocab_size]

        # === Advanced Frequency-Based Repetition Penalty ===
        # Strongly penalizes tokens that have appeared many times in generated text
        if idx.shape[1] > original_len:
            generated_so_far = idx[0, original_len:].tolist()
            token_counts = {}
            for t in generated_so_far:
                token_counts[t] = token_counts.get(t, 0) + 1

            for token, count in token_counts.items():
                # Quadratic penalty + extra for very frequent tokens
                penalty = 1.0 + (count ** 2) * 0.08
                if count > 4:
                    penalty += 1.5  # Strong kick for heavy repeats
                logits[0, token] /= max(1.1, penalty)

        # === Top-p (Nucleus) Sampling ===
        if TOP_P < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens above top_p
            sorted_indices_to_remove = cumulative_probs > TOP_P
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("inf")

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx


def main():
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    print("Initializing model...")
    model = My_GPT_model(
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

    # Prompt
    # prompt = """
    # मैंने बैंक से 5 लाख का पर्सनल लोन लिया है। अब EMI नहीं भर पा रहा। बैंक वाले धमकी दे रहे हैं। 
    # क्या करूँ कि लोन माफ हो जाए या EMI कम हो?
    # """

#     prompt = """

#     30 मार्च 2015 को, यह घोषणा की गई कि बेयोंस एक सह-स्वामी है, विभिन्न अन्य संगीत कलाकारों के साथ, संगीत स्ट्रीमिंग सेवा ज्वारीय में । यह सेवा हानिरहित ऑडियो और उच्च परिभाषा संगीत वीडियो में माहिर है । बेयोंस के पति जे जेड ने ज्वारीय और जे-जेड, सोलह कलाकार हितधारकों (जैसे कान्ये वेस्ट, रिहाना, मैडोना, क्रिस मार्टिन, निकी मिनाज और अधिक) सहित 2015. की पहली तिमाही में ज्वारीय, aspiro की जनक कंपनी का अधिग्रहण किया । सह-स्वयं ज्वारीय, बहुमत के साथ 3 % इक्विटी हिस्सेदारी है । एक सभी कलाकारों के स्वामित्व वाली स्ट्रीमिंग सेवा होने का विचार उन लोगों द्वारा बनाया गया था जो वर्तमान संगीत उद्योग के भीतर स्ट्रीमिंग की बढ़ी मांग को अनुकूलित करने के लिए शामिल थे, और spotify जैसे अन्य स्ट्रीमिंग सेवाओं के लिए, जो उनके कम भुगतान के लिए आलोचना की गई है रॉयल्टी का । चुनौती है कि हर किसी को फिर से संगीत का सम्मान करने के लिए, अपने मूल्य को पहचानने के लिए, ज्वारीय की रिहाई पर जे-जेड कहा गया ।
# ज्वारीय की जनक कंपनी 2015 में किसके स्वामित्व में बनी?

        # """

    # prompt = """### Instruction:
    # नमस्ते

    # ### Response:
    # """

    # prompt = "नमस्ते"
        
    # prompt = "आकाशगंगा के सबसे दूर तारे पर बसे छोटे कछुओं की भाषा समझना कठिन था। ये कहानी शुरू होती है..."
    # prompt = """एक जंगल में तीन जानवर थे:
    #     1. बोलने वाला कछुआ
    #     2. गणित जानने वाला कौआ
    #     3. झूठ बोलने वाला शेर

    #     कहानी में तीनों का उपयोग होना चाहिए।
    #     कहानी शुरू होती है:"""
    # तुम यहाँ कोई भी प्रॉम्प्ट बदल सकते हो
    # prompt = "महात्मा गांधी के बारे में पांच वाक्य लिखो:"
    prompt = (
        "### प्रश्न:\n"
        "महात्मा गांधी के बारे में पांच वाक्य लिखो:\n\n"
        "### उत्तर:\n"
    )
    
    print(f"Prompt: {prompt}")

    input_ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([input_ids], device=DEVICE)

    print("\nGenerating 5 different completions...\n")

    for i in range(3):
        print(f"--- Generation {i+1}/3 ---")
        output_ids = generate(model, x.clone(), MAX_NEW_TOKENS)
        generated_ids = output_ids[0, len(input_ids):].tolist()
        generated_text = sp.decode(generated_ids)
        print(generated_text)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()




