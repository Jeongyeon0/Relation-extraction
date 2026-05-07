def check_max_length_for_d_model_safe(
    train_path="refined-klue-re/train.tsv",
    dev_path="refined-klue-re/dev.tsv",
    model_name="klue/bert-base",
    label_desc_path=None,
    max_length_candidates=(128, 256, 384, 512),
    num_examples_to_show=5,
):
    import os
    import json
    import pandas as pd
    from transformers import AutoTokenizer

    SUBJ_MARKERS = ["[SUBJ-PER]", "[SUBJ-ORG]", "[SUBJ-LOC]",]

    OBJ_MARKERS = [
        "[OBJ-PER]",
        "[OBJ-ORG]",
        "[OBJ-LOC]",
        "[OBJ-POH]",
        "[OBJ-DAT]",
        "[OBJ-NOH]",
    ]

    ENTITY_SPECIAL_TOKENS = [
        "[SUBJ-PER]", "[/SUBJ-PER]",
        "[SUBJ-ORG]", "[/SUBJ-ORG]",
        "[SUBJ-LOC]", "[/SUBJ-LOC]",
        "[OBJ-PER]", "[/OBJ-PER]",
        "[OBJ-ORG]", "[/OBJ-ORG]",
        "[OBJ-LOC]", "[/OBJ-LOC]",
        "[OBJ-POH]", "[/OBJ-POH]",
        "[OBJ-DAT]", "[/OBJ-DAT]",
        "[OBJ-NOH]", "[/OBJ-NOH]",
    ]

    REL_TOKEN = "[REL]"
    SPECIAL_TOKENS = ENTITY_SPECIAL_TOKENS + [REL_TOKEN]

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

    def build_label_map(train_df, dev_df=None):
        labels = set(train_df["label"].unique())

        if dev_df is not None:
            labels = labels | set(dev_df["label"].unique())

        labels = list(labels)

        if "no_relation" in labels:
            labels.remove("no_relation")
            labels = ["no_relation"] + sorted(labels)
        else:
            labels = sorted(labels)

        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        return label2id, id2label

    def load_label_texts(label_desc_path, id2label):
        # description을 쓰지 않으면 label name만 사용
        if label_desc_path is None:
            return {idx: label for idx, label in id2label.items()}

        with open(label_desc_path, "r", encoding="utf-8") as f:
            desc_dict = json.load(f)

        label_texts = {}

        for idx, label in id2label.items():
            if label in desc_dict:
                label_texts[idx] = f"{label}: {desc_dict[label]}"
            else:
                label_texts[idx] = label

        return label_texts

    def make_candidate_text(id2label, label_texts):
        chunks = []

        for idx in range(len(id2label)):
            chunks.append(f"{REL_TOKEN} {label_texts[idx]}")

        return " ".join(chunks)

    train_df = read_tsv(train_path)
    dev_df = read_tsv(dev_path)

    label2id, id2label = build_label_map(train_df, dev_df)
    label_texts = load_label_texts(label_desc_path, id2label)

    candidate_text = make_candidate_text(id2label, label_texts)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS
    })

    rel_token_id = tokenizer.convert_tokens_to_ids(REL_TOKEN)

    subj_marker_ids = set(
        tokenizer.convert_tokens_to_ids(marker)
        for marker in SUBJ_MARKERS
    )

    obj_marker_ids = set(
        tokenizer.convert_tokens_to_ids(marker)
        for marker in OBJ_MARKERS
    )

    num_labels = len(label2id)

    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            dev_df.assign(split="dev"),
        ],
        ignore_index=True
    )

    candidate_len = len(
        tokenizer(
            candidate_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
    )

    print("=" * 80)
    print("Basic information")
    print("=" * 80)
    print(f"Number of labels: {num_labels}")
    print(f"Candidate token length: {candidate_len}")
    print(f"Candidate text preview:\n{candidate_text[:500]}...\n")

    max_sentence_len = 0
    max_total_len = 0

    for _, row in all_df.iterrows():
        sentence = row["sentence"]

        sentence_len = len(
            tokenizer(
                sentence,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
        )

        # BERT pair input 기준 대략:
        # [CLS] sentence [SEP] candidate [SEP]
        total_len = sentence_len + candidate_len + 3

        max_sentence_len = max(max_sentence_len, sentence_len)
        max_total_len = max(max_total_len, total_len)

    print(f"Max sentence token length: {max_sentence_len}")
    print(f"Max total token length without truncation: {max_total_len}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print()

    results = []

    for max_length in max_length_candidates:
        total = len(all_df)
        ok = 0
        fail = 0

        fail_examples = []

        for _, row in all_df.iterrows():
            sentence = row["sentence"]

            try:
                encoded = tokenizer(
                    sentence,
                    candidate_text,
                    truncation="only_second",
                    max_length=max_length,
                    padding="max_length",
                    return_tensors=None,
                )
            except Exception as e:
                fail += 1

                if len(fail_examples) < num_examples_to_show:
                    fail_examples.append({
                        "split": row["split"],
                        "guid": row["guid"],
                        "reason": str(e),
                        "sentence": sentence[:200],
                    })

                continue

            input_ids = encoded["input_ids"]

            rel_count = sum(1 for token_id in input_ids if token_id == rel_token_id)
            has_subj = any(token_id in subj_marker_ids for token_id in input_ids)
            has_obj = any(token_id in obj_marker_ids for token_id in input_ids)

            is_ok = (rel_count == num_labels) and has_subj and has_obj

            if is_ok:
                ok += 1
            else:
                fail += 1

                if len(fail_examples) < num_examples_to_show:
                    fail_examples.append({
                        "split": row["split"],
                        "guid": row["guid"],
                        "reason": f"rel_count={rel_count}, expected={num_labels}, has_subj={has_subj}, has_obj={has_obj}",
                        "sentence": sentence[:200],
                    })

        results.append({
            "max_length": max_length,
            "ok": ok,
            "fail": fail,
            "total": total,
            "ok_ratio": ok / total,
            "fail_examples": fail_examples,
        })

    print("=" * 80)
    print("Max length check summary")
    print("=" * 80)

    for result in results:
        print(
            f"max_length={result['max_length']} | "
            f"ok={result['ok']}/{result['total']} | "
            f"fail={result['fail']} | "
            f"ok_ratio={result['ok_ratio']:.4f}"
        )

    valid_lengths = [
        result["max_length"]
        for result in results
        if result["fail"] == 0
    ]

    print()
    print("=" * 80)
    print("Recommended max_length")
    print("=" * 80)

    if valid_lengths:
        print(f"Recommended max_length: {min(valid_lengths)}")
    else:
        print("No tested max_length fully preserves all relation candidates.")
        print("You need to shorten label descriptions, use label names only, or split candidate labels into chunks.")

    print()
    print("=" * 80)
    print("Failure examples")
    print("=" * 80)

    for result in results:
        if result["fail"] == 0:
            continue

        print(f"\n[max_length={result['max_length']}]")
        for ex in result["fail_examples"]:
            print(f"- {ex['split']} | {ex['guid']}")
            print(f"  reason: {ex['reason']}")
            print(f"  sentence: {ex['sentence']}")

    return results


results = check_max_length_for_d_model_safe(
    train_path="refined-klue-re/train.tsv",
    dev_path="refined-klue-re/dev.tsv",
    model_name="klue/bert-base",
    label_desc_path=None,
    max_length_candidates=(128, 256, 384, 512),
)   