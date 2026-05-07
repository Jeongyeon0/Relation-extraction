import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_tsv(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                raise ValueError(f"TSV format error in {path}: {line}")

            guid, sentence, label = parts

            rows.append({
                "guid": guid,
                "sentence": sentence,
                "label": label,
            })

    return pd.DataFrame(rows)


def write_tsv(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            line = f"{row['guid']}\t{row['sentence']}\t{row['label']}\n"
            f.write(line)


def split_train_valid(
    input_path="refined-klue-re/train.tsv",
    output_dir="refined-klue-re",
    train_output_name="train_split.tsv",
    valid_output_name="vaild_split.tsv",
    valid_size=0.1,
    seed=42,
):
    df = read_tsv(input_path)

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
        stratify=df["label"],
    )

    os.makedirs(output_dir, exist_ok=True)

    train_output_path = os.path.join(output_dir, train_output_name)
    valid_output_path = os.path.join(output_dir, valid_output_name)

    write_tsv(train_df, train_output_path)
    write_tsv(valid_df, valid_output_path)

    print("Split finished.")
    print(f"Input: {input_path}")
    print(f"Train: {train_output_path} ({len(train_df)} samples)")
    print(f"Valid: {valid_output_path} ({len(valid_df)} samples)")

    print("\nLabel distribution - original")
    print(df["label"].value_counts(normalize=True).sort_index())

    print("\nLabel distribution - train_split")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nLabel distribution - vaild_split")
    print(valid_df["label"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    split_train_valid(
        input_path="refined-klue-re/train.tsv",
        output_dir="refined-klue-re",
        train_output_name="train_split.tsv",
        valid_output_name="valid_split.tsv",
        valid_size=0.1,
        seed=42,
    )