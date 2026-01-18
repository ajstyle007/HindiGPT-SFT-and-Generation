import gradio as gr
import torch, re
import sentencepiece as spm
from sft_gen import generate
from decoder_only_gpt import My_GPT_model

# ------------------ Load tokenizer ------------------
sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Load model ------------------
model = My_GPT_model(
    vocab_size=sp.get_piece_size(),
    num_layers=12,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    seq_len=512
).to(DEVICE)

model.load_state_dict(torch.load("full_sft_final.pt", map_location=DEVICE))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# ------------------ Helpers ------------------
def encode_text(text, max_len=512):
    ids = sp.encode(text, out_type=int)[:max_len]
    return torch.tensor([ids], device=DEVICE)

def decode_tokens(token_ids):
    return sp.decode(token_ids[0].tolist())

def post_clean(text):
    text = text.replace("⁇", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


sample_questions = [
["इटरनेट कैसे काम करता है?"],
["लोकतंत्र क्या है?"],
["गुनाहों का देवता उपन्यास किसने लिखा?"],
["मशीन लर्निंग क्या है?"],
["1857 की क्रांति क्या थी?"],
["भारत की राजधानी क्या है?"],
["एक रचनात्मक कहानी लिखिए।"],
["अगर आपको बिना इंटरनेट के 24 घंटे बिताने पड़ें, तो आप उस समय क्या-क्या करेंगे?"],
["अगर AI इंसानों की तरह सोचने लगे, तो सबसे पहले कौन-सी चीज़ बदलेगी?"],
["क्या पैसा खुशी खरीद सकता है?"],
["सफलता भाग्य से मिलती है या मेहनत से?"],
["क्या इंसान कभी अमर हो सकता है?"],
["क्या ब्रह्मांड पूरी तरह पूर्व-निर्धारित है?"],
["क्या समय एक भ्रम है?"],
["भगवान को किसने बनाया?"],
["अगर भगवान मर गया, तो नैतिकता किसकी होगी?"]
]

# ------------------ Gradio function ------------------
@torch.no_grad()
def gradio_wrapper(query, top_k, max_tokens, temperature, top_p):
    if not query.strip():
        return "कृपया प्रश्न लिखें।"

    prompt = f"### प्रश्न:\n{query}\n\n### उत्तर:\n"


    input_ids = encode_text(prompt)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # safety clamps (IMPORTANT)
        top_k = max(20, min(int(top_k), 80))
        max_tokens = max(50, min(int(max_tokens), 600))

        temperature = max(0.2, min(float(temperature), 1.0))
        top_p = max(0.6, min(float(top_p), 0.95))

        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.12,
            eos_token_id=sp.eos_id(),
            pad_token_id=sp.bos_id()
        )

    answer = decode_tokens(output_ids)
    answer = post_clean(answer)

    # Optional: sirf answer part return karo
    if "### उत्तर:" in answer:
        answer = answer.split("### उत्तर:")[-1].strip()

    return answer


custom_css = """
.gradio-container {
    font-family: Arial, sans-serif;
    background-color: #1e1e1e;
    color: white;
    padding: 20px;
}
.label {
    color: orange !important;
    font-weight: bold;
}

.gr-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 15px;
}
.gr-row > div {
    flex: 1 1 200px;
    max-width: 220px;
}

.submit-btn {
    background-color: #FF5F1F !important;
    color: white !important;
    font-weight: bold;
}

.submit-btn:hover {
    background-color: #FF5F1F !important;
    transform: scale(1.03);
}

.submit-btn {
    cursor: pointer;
}

.param-hint {
    margin-top: 3px;
    margin-bottom: 2px;
    font-size: 12px;              /* smaller, professional */
    line-height: 1.4;
    color: #a8a8a8;               /* softer than pure white */
    background: rgba(255, 255, 255, 0.03);
    padding: 6px 10px;
    border-left: 2px solid #FF5F1F;
    border-radius: 6px;
}

"""



def main():

    with gr.Blocks(css=custom_css) as demo:

        gr.Markdown(
            """
            <h1 style="
                text-align:center;
                margin-top:40px;
                font-size:clamp(25px, 3vw, 40px);
                font-weight:700;
                color:#ffffff;
            ">
            ❓ हिंदी GPT<span style="color:#0EA5E9;"> प्रश्न-उत्तर</span>
            </h1>
            <p style="color:gray; text-align:center; margin:25px 0 13px 0;">
            A Hindi GPT model built from scratch and fine-tuned on curated हिंदी Q&A data
            </p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(lines=3, placeholder="अपना प्रश्न यहाँ लिखें...", label="प्रश्न")

                with gr.Row():
                    gr.Markdown(
                            """
                            <div class="param-hint">
                            Default parameters are optimized for high-quality output.
                            Adjust them to control response length, diversity, and creativity.
                            </div>
                            """
                        )


                with gr.Row():
                    top_k = gr.Dropdown(choices=[20,30,45,50,60,70,80], value=45, label="Top-K")
                    max_tokens = gr.Dropdown(choices=[50,100,200,300,400], value=300, label="Max Tokens")
                    temperature = gr.Dropdown(choices=[0.2,0.4,0.6,0.7,0.85,1.0], value=0.85, label="Temperature")
                    top_p = gr.Dropdown(choices=[0.6,0.7,0.8,0.92,0.95], value=0.92, label="Top-P")

                with gr.Row():
                    clear_btn  = gr.Button("Clear", elem_classes="clear-btn")
                    submit_btn = gr.Button("Generate", elem_classes="submit-btn")
                    

            with gr.Column(scale=1):
                output = gr.Textbox(lines=15, label="उत्तर")

        examples = gr.Examples(
                examples=sample_questions,
                inputs=[query, top_k, max_tokens, temperature, top_p],
                label="उदाहरण प्रश्न (Click on these examples for automatic input.)"
            )
        
        gr.Markdown(
            """
            <details class="param-details">
            <summary><b>Parameter Overview</b></summary>
            <ul>
                <li><b>Top-K</b>: Limits token selection to the top K most probable options.</li>
                <li><b>Max Tokens</b>: Maximum number of tokens generated in the response.</li>
                <li><b>Temperature</b>: Controls randomness (lower = more deterministic).</li>
                <li><b>Top-P</b>: Enables nucleus sampling by selecting from the smallest probability mass.</li>
            </ul>
            </details>
            """
        )

        gr.Markdown(
            """
            <div style="text-align:center; font-size:12px; color:#9a9a9a;">
            A custom GPT architecture developed from scratch by Ajay Kumar, incorporating a domain-specific tokenizer and fine-tuned on Hindi Q&A pairs for controlled text generation.  
            The model consists of approximately 57.7 million trainable parameters.
            </div>
            """
        )

                

        # Submit button click will call the inference 
        submit_btn.click(
            gradio_wrapper,
            inputs=[query, top_k, max_tokens, temperature, top_p],
            outputs=output
        )

        # submit button
        query.submit(
            gradio_wrapper,
            inputs=[query, top_k, max_tokens, temperature, top_p],
            outputs=output
        )

        # Clear button will clear the inputs and output
        def clear_all():
            return "", top_k, max_tokens ,temperature, top_p, ""

        clear_btn.click(
            clear_all,
            inputs=[],
            outputs=[query, top_k, max_tokens, temperature, top_p, output]
        )

    demo.launch()


if __name__ == "__main__":
    main()

