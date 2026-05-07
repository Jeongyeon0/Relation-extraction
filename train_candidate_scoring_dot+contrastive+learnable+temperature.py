import os
import json
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm


# =========================================================
# 1. 기본 설정
# =========================================================

ENTITY_TYPES = ["PER", "ORG", "LOC", "POH", "DAT", "NOH"]

SUBJ_MARKERS = [f"[SUBJ-{t}]" for t in ENTITY_TYPES]
OBJ_MARKERS = [f"[OBJ-{t}]" for t in ENTITY_TYPES]

ENTITY_SPECIAL_TOKENS = []

for t in ENTITY_TYPES:
    ENTITY_SPECIAL_TOKENS.extend([
        f"[SUBJ-{t}]",
        f"[/SUBJ-{t}]",
        f"[OBJ-{t}]",
        f"[/OBJ-{t}]",
    ])

REL_TOKEN = "[REL]"
SPECIAL_TOKENS = ENTITY_SPECIAL_TOKENS + [REL_TOKEN]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. TSV 로드
#    형식: guid \t sentence \t label
# =========================================================

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


def load_label_descriptions(label_desc_path, id2label):
    """
    label_desc_path가 있으면 JSON에서 label 설명을 불러옴.
    없으면 label name 자체를 후보 텍스트로 사용.

    description 파일이 있을 경우 label name을 제거하고
    description만 relation 후보 텍스트로 사용함.
    """
    if label_desc_path is None:
        return {idx: label for idx, label in id2label.items()}

    if not os.path.exists(label_desc_path):
        raise FileNotFoundError(f"Cannot find label description file: {label_desc_path}")

    with open(label_desc_path, "r", encoding="utf-8") as f:
        desc_dict = json.load(f)

    label_texts = {}

    for idx, label in id2label.items():
        if label in desc_dict:
            label_texts[idx] = desc_dict[label]
        else:
            label_texts[idx] = label

    return label_texts


def make_candidate_text(id2label, label_texts):
    chunks = []

    for idx in range(len(id2label)):
        chunks.append(f"{REL_TOKEN} {label_texts[idx]}")

    return " ".join(chunks)


# =========================================================
# 3. Dataset
#    relation 후보는 보존하고,
#    문장이 길면 subject/object marker 주변 window만 사용
# =========================================================

class RECandidateDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        label2id,
        id2label,
        label_texts,
        max_length=512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.label_texts = label_texts
        self.max_length = max_length
        self.num_labels = len(label2id)

        self.candidate_text = make_candidate_text(
            id2label=self.id2label,
            label_texts=self.label_texts,
        )

        self.subj_marker_ids = set(
            tokenizer.convert_tokens_to_ids(marker)
            for marker in SUBJ_MARKERS
        )

        self.obj_marker_ids = set(
            tokenizer.convert_tokens_to_ids(marker)
            for marker in OBJ_MARKERS
        )

        self.rel_token_id = tokenizer.convert_tokens_to_ids(REL_TOKEN)

        self.candidate_ids = tokenizer(
            self.candidate_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]

        # [CLS] sentence [SEP] candidate [SEP]
        self.num_pair_special_tokens = 3

        self.max_sentence_len = (
            self.max_length
            - len(self.candidate_ids)
            - self.num_pair_special_tokens
        )

        if self.max_sentence_len <= 0:
            raise ValueError(
                "Relation candidate text is too long. "
                f"candidate_len={len(self.candidate_ids)}, "
                f"max_length={self.max_length}. "
                "Shorten label descriptions or use label names only."
            )

        print("=" * 80)
        print("RECandidateDataset initialized")
        print("=" * 80)
        print(f"num_labels: {self.num_labels}")
        print(f"candidate_len: {len(self.candidate_ids)}")
        print(f"max_length: {self.max_length}")
        print(f"max_sentence_len: {self.max_sentence_len}")

    def __len__(self):
        return len(self.df)

    def find_marker_position_in_sentence_ids(self, sentence_ids, marker_id_set, sentence):
        for idx, token_id in enumerate(sentence_ids):
            if token_id in marker_id_set:
                return idx

        raise ValueError(f"Entity marker not found in sentence tokens: {sentence}")

    def make_marker_preserved_sentence_window(self, sentence_ids, subj_pos, obj_pos, sentence):
        if len(sentence_ids) <= self.max_sentence_len:
            return sentence_ids

        left_marker = min(subj_pos, obj_pos)
        right_marker = max(subj_pos, obj_pos)

        required_span_len = right_marker - left_marker + 1

        if required_span_len > self.max_sentence_len:
            raise ValueError(
                "Cannot preserve both subject and object markers within max_sentence_len. "
                f"required_span_len={required_span_len}, "
                f"max_sentence_len={self.max_sentence_len}\n"
                f"Sentence: {sentence}"
            )

        remaining = self.max_sentence_len - required_span_len
        left_context = remaining // 2
        right_context = remaining - left_context

        window_start = left_marker - left_context
        window_end = right_marker + right_context + 1

        if window_start < 0:
            window_end += -window_start
            window_start = 0

        if window_end > len(sentence_ids):
            overflow = window_end - len(sentence_ids)
            window_start = max(0, window_start - overflow)
            window_end = len(sentence_ids)

        sentence_window_ids = sentence_ids[window_start:window_end]

        has_subj = any(token_id in self.subj_marker_ids for token_id in sentence_window_ids)
        has_obj = any(token_id in self.obj_marker_ids for token_id in sentence_window_ids)

        if not has_subj or not has_obj:
            raise ValueError(
                "Subject or object marker was lost during sentence window truncation.\n"
                f"has_subj={has_subj}, has_obj={has_obj}\n"
                f"Sentence: {sentence}"
            )

        return sentence_window_ids

    def build_pair_inputs(self, sentence_ids):
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        if cls_id is None or sep_id is None or pad_id is None:
            raise ValueError(
                "Tokenizer must have cls_token_id, sep_token_id, and pad_token_id."
            )

        input_ids = (
            [cls_id]
            + sentence_ids
            + [sep_id]
            + self.candidate_ids
            + [sep_id]
        )

        attention_mask = [1] * len(input_ids)

        token_type_ids = (
            [0] * (len(sentence_ids) + 2)
            + [1] * (len(self.candidate_ids) + 1)
        )

        if len(input_ids) > self.max_length:
            raise ValueError(
                "Input is still longer than max_length after sentence windowing. "
                f"len(input_ids)={len(input_ids)}, "
                f"max_length={self.max_length}, "
                f"sentence_len={len(sentence_ids)}, "
                f"candidate_len={len(self.candidate_ids)}"
            )

        pad_len = self.max_length - len(input_ids)

        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def find_marker_position(self, input_ids, marker_id_set, sentence):
        for idx, token_id in enumerate(input_ids):
            if token_id in marker_id_set:
                return idx

        raise ValueError(f"Entity marker not found after final input construction: {sentence}")

    def find_rel_positions(self, input_ids, sentence):
        rel_positions = []

        for idx, token_id in enumerate(input_ids):
            if token_id == self.rel_token_id:
                rel_positions.append(idx)

        if len(rel_positions) != self.num_labels:
            raise ValueError(
                "The number of [REL] markers is not equal to the number of labels. "
                f"Found {len(rel_positions)}, expected {self.num_labels}. "
                "Relation candidates should not be truncated.\n"
                f"Sentence: {sentence}"
            )

        return rel_positions

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        guid = item["guid"]
        sentence = item["sentence"]
        label = item["label"]

        sentence_ids = self.tokenizer(
            sentence,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]

        subj_pos_in_sentence = self.find_marker_position_in_sentence_ids(
            sentence_ids,
            self.subj_marker_ids,
            sentence,
        )

        obj_pos_in_sentence = self.find_marker_position_in_sentence_ids(
            sentence_ids,
            self.obj_marker_ids,
            sentence,
        )

        sentence_window_ids = self.make_marker_preserved_sentence_window(
            sentence_ids=sentence_ids,
            subj_pos=subj_pos_in_sentence,
            obj_pos=obj_pos_in_sentence,
            sentence=sentence,
        )

        encoded = self.build_pair_inputs(sentence_window_ids)
        input_ids = encoded["input_ids"]

        subj_pos = self.find_marker_position(
            input_ids=input_ids,
            marker_id_set=self.subj_marker_ids,
            sentence=sentence,
        )

        obj_pos = self.find_marker_position(
            input_ids=input_ids,
            marker_id_set=self.obj_marker_ids,
            sentence=sentence,
        )

        rel_positions = self.find_rel_positions(
            input_ids=input_ids,
            sentence=sentence,
        )

        result = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "subj_pos": torch.tensor(subj_pos, dtype=torch.long),
            "obj_pos": torch.tensor(obj_pos, dtype=torch.long),
            "rel_positions": torch.tensor(rel_positions, dtype=torch.long),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),

            "guid": guid,
            "sentence": sentence,
            "label_text": label,
        }

        result["token_type_ids"] = torch.tensor(
            encoded["token_type_ids"],
            dtype=torch.long
        )

        return result


# =========================================================
# 4. dot-only + learnable temperature 모델
#    score = dot-product score / temperature
# =========================================================

class RECandidateScoringModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
        dropout=0.1,
        init_temperature=0.05,
        min_temperature=1e-4,
        max_temperature=10.0,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self.pair_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive.")

        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float)
        )

    def resize_token_embeddings(self, tokenizer):
        self.encoder.resize_token_embeddings(len(tokenizer))

    def get_temperature(self):
        return torch.exp(self.log_temperature).clamp(
            min=self.min_temperature,
            max=self.max_temperature,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        subj_pos,
        obj_pos,
        rel_positions,
        token_type_ids=None,
    ):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # BERT는 token_type_ids 사용 가능.
        # RoBERTa는 type_vocab_size가 보통 1이므로 token_type_ids를 넘기지 않음.
        type_vocab_size = getattr(self.encoder.config, "type_vocab_size", 0)
        if token_type_ids is not None and type_vocab_size > 1:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        hidden = outputs.last_hidden_state

        batch_size = hidden.size(0)
        batch_indices = torch.arange(batch_size, device=hidden.device)

        h_cls = hidden[:, 0, :]
        h_subj = hidden[batch_indices, subj_pos, :]
        h_obj = hidden[batch_indices, obj_pos, :]

        pair_features = torch.cat(
            [
                h_cls,
                h_subj,
                h_obj,
                h_subj * h_obj,
                torch.abs(h_subj - h_obj),
            ],
            dim=-1
        )

        h_pair = self.pair_proj(pair_features)

        expanded_batch_indices = batch_indices.unsqueeze(1).expand(
            batch_size,
            rel_positions.size(1)
        )

        h_rel = hidden[expanded_batch_indices, rel_positions, :]

        h_pair_expanded = h_pair.unsqueeze(1).expand_as(h_rel)

        dot_scores = torch.sum(
            h_pair_expanded * h_rel,
            dim=-1
        )

        temperature = self.get_temperature()
        scores = dot_scores / temperature

        return {
            "scores": scores,
            "dot_scores": dot_scores,
            "temperature": temperature,
        }


# =========================================================
# 5. Loss weight
# =========================================================

def compute_class_weights(train_df, label2id):
    counts = train_df["label"].value_counts().to_dict()
    num_labels = len(label2id)
    total = len(train_df)

    weights = torch.ones(num_labels, dtype=torch.float)

    for label, idx in label2id.items():
        count = counts.get(label, 1)
        weights[idx] = total / (num_labels * count)

    return weights


# =========================================================
# 6. 평가 + prediction 저장
# =========================================================

def compute_auprc_metrics(all_labels, all_probs, id2label, no_relation_id=0):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_labels = len(id2label)

    y_true = np.zeros((len(all_labels), num_labels), dtype=int)
    y_true[np.arange(len(all_labels)), all_labels] = 1

    def safe_macro_auprc(label_ids):
        scores = []
        per_label = {}

        for label_id in label_ids:
            if y_true[:, label_id].sum() == 0:
                per_label[id2label[label_id]] = None
                continue

            auprc = average_precision_score(
                y_true[:, label_id],
                all_probs[:, label_id],
            )

            scores.append(auprc)
            per_label[id2label[label_id]] = auprc

        if len(scores) == 0:
            return 0.0, per_label

        return float(np.mean(scores)), per_label

    all_label_ids = list(range(num_labels))
    positive_label_ids = [
        label_id for label_id in all_label_ids
        if label_id != no_relation_id
    ]

    macro_auprc, per_label_auprc = safe_macro_auprc(all_label_ids)

    macro_auprc_without_no_relation, per_label_auprc_without_no_relation = safe_macro_auprc(
        positive_label_ids
    )

    micro_auprc = average_precision_score(
        y_true,
        all_probs,
        average="micro",
    )

    return {
        "macro_auprc": macro_auprc,
        "micro_auprc": float(micro_auprc),
        "macro_auprc_without_no_relation": macro_auprc_without_no_relation,
        "per_label_auprc": per_label_auprc,
        "per_label_auprc_without_no_relation": per_label_auprc_without_no_relation,
    }


def evaluate(
    model,
    dataloader,
    device,
    id2label,
    no_relation_id=0,
    output_prediction_path=None,
):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    prediction_rows = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            guids = batch.pop("guid")
            sentences = batch.pop("sentence")
            label_texts = batch.pop("label_text")

            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch.pop("labels")
            outputs = model(**batch)
            scores = outputs["scores"]

            probs = torch.softmax(scores, dim=-1)
            preds = torch.argmax(scores, dim=-1)

            pred_ids = preds.cpu().numpy().tolist()
            gold_ids = labels.cpu().numpy().tolist()
            prob_values = probs.cpu().numpy()

            all_preds.extend(pred_ids)
            all_labels.extend(gold_ids)
            all_probs.extend(prob_values)

            for guid, sentence, gold_label_text, pred_id, prob in zip(
                guids,
                sentences,
                label_texts,
                pred_ids,
                prob_values,
            ):
                pred_label_text = id2label[pred_id]
                pred_score = float(prob[pred_id])

                prediction_rows.append({
                    "guid": guid,
                    "sentence": sentence,
                    "gold_label": gold_label_text,
                    "pred_label": pred_label_text,
                    "pred_score": pred_score,
                })

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")

    filtered_labels = []
    filtered_preds = []

    for gold, pred in zip(all_labels, all_preds):
        if gold != no_relation_id:
            filtered_labels.append(gold)
            filtered_preds.append(pred)

    if len(filtered_labels) > 0:
        micro_f1_without_no_relation = f1_score(
            filtered_labels,
            filtered_preds,
            average="micro"
        )
    else:
        micro_f1_without_no_relation = 0.0

    auprc_metrics = compute_auprc_metrics(
        all_labels=all_labels,
        all_probs=all_probs,
        id2label=id2label,
        no_relation_id=no_relation_id,
    )

    target_names = [id2label[i] for i in range(len(id2label))]

    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    if output_prediction_path is not None:
        os.makedirs(os.path.dirname(output_prediction_path), exist_ok=True)

        with open(output_prediction_path, "w", encoding="utf-8") as f:
            for row in prediction_rows:
                line = (
                    f"{row['guid']}\t"
                    f"{row['sentence']}\t"
                    f"{row['gold_label']}\t"
                    f"{row['pred_label']}\t"
                    f"{row['pred_score']:.6f}\n"
                )
                f.write(line)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_f1_without_no_relation": micro_f1_without_no_relation,

        "macro_auprc": auprc_metrics["macro_auprc"],
        "micro_auprc": auprc_metrics["micro_auprc"],
        "macro_auprc_without_no_relation": auprc_metrics["macro_auprc_without_no_relation"],
        "per_label_auprc": auprc_metrics["per_label_auprc"],

        "report": report,
    }


# =========================================================
# 7. 학습
# =========================================================

def train(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = read_tsv(args.train_path)
    dev_df = read_tsv(args.dev_path)

    label2id, id2label = build_label_map(train_df, dev_df)
    label_texts = load_label_descriptions(args.label_desc_path, id2label)

    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "label_texts.json"), "w", encoding="utf-8") as f:
        json.dump(label_texts, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS
    })

    train_dataset = RECandidateDataset(
        df=train_df,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        label_texts=label_texts,
        max_length=args.max_length,
    )

    dev_dataset = RECandidateDataset(
        df=dev_df,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        label_texts=label_texts,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = RECandidateScoringModel(
        model_name=args.model_name,
        num_labels=len(label2id),
        dropout=args.dropout,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )

    model.resize_token_embeddings(tokenizer)
    model.to(device)

    no_relation_id = label2id.get("no_relation", 0)

    if args.use_class_weight:
        class_weights = compute_class_weights(train_df, label2id).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress, start=1):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            labels = batch.pop("labels")

            batch.pop("guid", None)
            batch.pop("sentence", None)
            batch.pop("label_text", None)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                scores = outputs["scores"]
                loss = criterion(scores, labels)

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            current_temperature = model.get_temperature().detach().cpu().item()

            progress.set_postfix({
                "loss": total_loss / step,
                "temp": current_temperature,
            })

        avg_train_loss = total_loss / len(train_loader)
        current_temperature = model.get_temperature().detach().cpu().item()

        prediction_path = os.path.join(
            args.output_dir,
            f"dev_predictions_epoch_{epoch}.tsv"
        )

        metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            id2label=id2label,
            no_relation_id=no_relation_id,
            output_prediction_path=prediction_path,
        )

        print(f"\nEpoch {epoch}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Learned temperature: {current_temperature:.6f}")
        print(f"Dev accuracy: {metrics['accuracy']:.4f}")
        print(f"Dev macro F1: {metrics['macro_f1']:.4f}")
        print(f"Dev micro F1: {metrics['micro_f1']:.4f}")
        print(f"Dev micro F1 without no_relation: {metrics['micro_f1_without_no_relation']:.4f}")

        print(f"Dev macro AUPRC: {metrics['macro_auprc']:.4f}")
        print(f"Dev micro AUPRC: {metrics['micro_auprc']:.4f}")
        print(f"Dev macro AUPRC without no_relation: {metrics['macro_auprc_without_no_relation']:.4f}")
        print(metrics["report"])
        print(f"Dev predictions saved to: {prediction_path}")

        score = metrics["micro_f1_without_no_relation"]

        if score > best_score:
            best_score = score
            patience_count = 0

            save_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(save_dir, exist_ok=True)

            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            torch.save(
                {
                    "pair_proj_state_dict": model.pair_proj.state_dict(),
                    "log_temperature": model.log_temperature.detach().cpu(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "label_texts": label_texts,
                    "args": vars(args),
                    "best_score": best_score,
                    "scoring_type": "dot_only_learnable_temperature",
                },
                os.path.join(save_dir, "model.pt")
            )

            best_prediction_path = os.path.join(
                args.output_dir,
                "best_dev_predictions.tsv"
            )

            with open(prediction_path, "r", encoding="utf-8") as src:
                with open(best_prediction_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

            print(f"Best model saved. Score: {best_score:.4f}")
            print(f"Best dev predictions saved to: {best_prediction_path}")

        else:
            patience_count += 1
            print(f"No improvement. Patience: {patience_count}/{args.patience}")

            if patience_count >= args.patience:
                print("Early stopping.")
                break

    print(f"Training finished. Best score: {best_score:.4f}")


# =========================================================
# 8. Evaluation-only mode
# =========================================================

def load_trained_model(args, device):
    checkpoint_path = os.path.join(args.eval_model_dir, "model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    label2id = checkpoint["label2id"]

    id2label = checkpoint["id2label"]
    id2label = {int(k): v for k, v in id2label.items()}

    label_texts = checkpoint["label_texts"]
    label_texts = {int(k): v for k, v in label_texts.items()}

    saved_args = checkpoint.get("args", {})

    init_temperature = saved_args.get("init_temperature", args.init_temperature)
    min_temperature = saved_args.get("min_temperature", args.min_temperature)
    max_temperature = saved_args.get("max_temperature", args.max_temperature)

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_dir)

    model = RECandidateScoringModel(
        model_name=args.eval_model_dir,
        num_labels=len(label2id),
        dropout=args.dropout,
        init_temperature=init_temperature,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )

    model.resize_token_embeddings(tokenizer)

    model.pair_proj.load_state_dict(checkpoint["pair_proj_state_dict"])

    if "log_temperature" in checkpoint:
        model.log_temperature.data.copy_(
            checkpoint["log_temperature"].to(model.log_temperature.device)
        )

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint scoring_type: {checkpoint.get('scoring_type')}")
    print(f"Loaded checkpoint best_score: {checkpoint.get('best_score')}")
    print(f"Loaded temperature: {model.get_temperature().detach().cpu().item():.6f}")

    return model, tokenizer, label2id, id2label, label_texts


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer, label2id, id2label, label_texts = load_trained_model(
        args,
        device,
    )

    eval_df = read_tsv(args.eval_path)

    eval_dataset = RECandidateDataset(
        df=eval_df,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        label_texts=label_texts,
        max_length=args.max_length,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    no_relation_id = label2id.get("no_relation", 0)

    os.makedirs(args.output_dir, exist_ok=True)

    output_prediction_path = os.path.join(
        args.output_dir,
        args.eval_output_name,
    )

    metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        device=device,
        id2label=id2label,
        no_relation_id=no_relation_id,
        output_prediction_path=output_prediction_path,
    )

    print("Evaluation finished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Micro F1 without no_relation: {metrics['micro_f1_without_no_relation']:.4f}")

    print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
    print(f"Micro AUPRC: {metrics['micro_auprc']:.4f}")
    print(f"Macro AUPRC without no_relation: {metrics['macro_auprc_without_no_relation']:.4f}")
    print(metrics["report"])
    print(f"Predictions saved to: {output_prediction_path}")


# =========================================================
# 9. Argument
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
    )

    parser.add_argument("--train_path", type=str, default="refined-klue-re/train.tsv")
    parser.add_argument("--dev_path", type=str, default="refined-klue-re/dev.tsv")
    parser.add_argument("--eval_path", type=str, default="refined-klue-re/dev.tsv")

    parser.add_argument("--output_dir", type=str, default="outputs/re_klue_roberta_base_dot_only_learnable_temp")
    parser.add_argument("--eval_model_dir", type=str, default="outputs/re_klue_roberta_base_dot_only_learnable_temp/best_model")
    parser.add_argument("--eval_output_name", type=str, default="eval_predictions.tsv")

    parser.add_argument("--model_name", type=str, default="klue/roberta-base")
    parser.add_argument("--label_desc_path", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--init_temperature", type=float, default=0.05)
    parser.add_argument("--min_temperature", type=float, default=1e-4)
    parser.add_argument("--max_temperature", type=float, default=10.0)

    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
