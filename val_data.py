# import pandas as pd

# # df = pd.read_parquet("validation-00000.parquet")

# df = pd.read_csv("Hindi_Test_ClosedDomainQA.csv")

# # print(df.head())
# print(df.columns)
# print(df.shape)
# print(df["question"][0])
# print(df["answer"][0])
# print(df[["question", "best_answer", "correct_answers"]])


import csv
import json

INPUT_CSV = "Hindi_Test_ClosedDomainQA.csv"
OUTPUT_JSONL = "hindi_sft_val.jsonl"

with open(INPUT_CSV, "r", encoding="utf-8") as f, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as out:

    reader = csv.DictReader(f)

    for row in reader:
        question = row["question"].strip()
        answer = row["answer"].strip()

        text = (
            "### प्रश्न:\n"
            f"{question}\n\n"
            "### उत्तर:\n"
            f"{answer}"
        )

        out.write(
            json.dumps({"text": text}, ensure_ascii=False) + "\n"
        )

print("✅ Conversion done →", OUTPUT_JSONL)


