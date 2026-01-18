import sentencepiece as spm

# sp = spm.SentencePieceProcessor()

# tokenizer = sp.Load("hindi_tokenizer_new.model")

# for i in range(50):  # first 50 tokens
#     print(i, sp.id_to_piece(i))

# print(sp.decode_ids([3, 485, 7890, 900]))

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="hindi_tokenizer_new.model")

print(sp.get_piece_size())

print("PAD:", sp.pad_id())
print("BOS:", sp.bos_id())
print("EOS:", sp.eos_id())
print("UNK:", sp.unk_id())