import os
import json
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, average_precision_score

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

SPECIAL_TOKENS = []
for entity_type in ENTITY_TYPES:
    SPECIAL_TOKENS.extend([
        f"[SUBJ-{entity_type}]",
        f"[/SUBJ-{entity_type}]",
        f"[OBJ-{entity_type}]",
        f"[/OBJ-{entity_type}]",
    ])

SUBJ_MARKERS = [f"[SUBJ-{entity_type}]" for entity_type in ENTITY_TYPES]
OBJ_MARKERS = [f"[OBJ-{entity_type}]" for entity_type in ENTITY_TYPES]


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
            rows.append({"guid": guid, "sentence": sentence, "label": label})
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

# =========================================================
# 3. Dataset
#    relation 후보는 보존하고,
#    문장이 길면 subject/object marker 주변 window만 사용
# =========================================================

class REDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.subj_marker_ids = set(tokenizer.convert_tokens_to_ids(marker) for marker in SUBJ_MARKERS)
        self.obj_marker_ids = set(tokenizer.convert_tokens_to_ids(marker) for marker in OBJ_MARKERS)
        self.num_special_tokens = 2  # [CLS] sentence [SEP]
        self.max_sentence_len = self.max_length - self.num_special_tokens
        if self.max_sentence_len <= 0:
            raise ValueError(f"max_length is too small: max_length={self.max_length}")

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
                f"required_span_len={required_span_len}, max_sentence_len={self.max_sentence_len}\n"
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
                f"has_subj={has_subj}, has_obj={has_obj}\nSentence: {sentence}"
            )
        return sentence_window_ids

    def build_single_inputs(self, sentence_ids):
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        if cls_id is None or sep_id is None or pad_id is None:
            raise ValueError("Tokenizer must have cls_token_id, sep_token_id, and pad_token_id.")
        input_ids = [cls_id] + sentence_ids + [sep_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        if len(input_ids) > self.max_length:
            raise ValueError(
                "Input is still longer than max_length after sentence windowing. "
                f"len(input_ids)={len(input_ids)}, max_length={self.max_length}, sentence_len={len(sentence_ids)}"
            )
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    def find_marker_position(self, input_ids, marker_id_set, sentence):
        for idx, token_id in enumerate(input_ids):
            if token_id in marker_id_set:
                return idx
        raise ValueError(f"Entity marker not found after final input construction: {sentence}")

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        guid = item["guid"]
        sentence = item["sentence"]
        label = item["label"]
        sentence_ids = self.tokenizer(sentence, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        subj_pos_in_sentence = self.find_marker_position_in_sentence_ids(sentence_ids, self.subj_marker_ids, sentence)
        obj_pos_in_sentence = self.find_marker_position_in_sentence_ids(sentence_ids, self.obj_marker_ids, sentence)
        sentence_window_ids = self.make_marker_preserved_sentence_window(sentence_ids, subj_pos_in_sentence, obj_pos_in_sentence, sentence)
        encoded = self.build_single_inputs(sentence_window_ids)
        input_ids = encoded["input_ids"]
        subj_pos = self.find_marker_position(input_ids, self.subj_marker_ids, sentence)
        obj_pos = self.find_marker_position(input_ids, self.obj_marker_ids, sentence)
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.long),
            "subj_pos": torch.tensor(subj_pos, dtype=torch.long),
            "obj_pos": torch.tensor(obj_pos, dtype=torch.long),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
            "guid": guid,
            "sentence": sentence,
            "label_text": label,
        }



# =========================================================
# 4. 모델: Label-aware classification
# =========================================================


class REClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, num_labels)

    def resize_token_embeddings(self, tokenizer):
        self.encoder.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, subj_pos, obj_pos, token_type_ids=None):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
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
        features = torch.cat([h_cls, h_subj, h_obj, h_subj * h_obj, torch.abs(h_subj - h_obj)], dim=-1)
        mlp_logits = self.mlp(features)
        bilinear_logits = self.bilinear(h_subj, h_obj)
        return mlp_logits + bilinear_logits


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
            auprc = average_precision_score(y_true[:, label_id], all_probs[:, label_id])
            scores.append(auprc)
            per_label[id2label[label_id]] = auprc
        if len(scores) == 0:
            return 0.0, per_label
        return float(np.mean(scores)), per_label

    all_label_ids = list(range(num_labels))
    positive_label_ids = [label_id for label_id in all_label_ids if label_id != no_relation_id]
    macro_auprc, per_label_auprc = safe_macro_auprc(all_label_ids)
    macro_auprc_without_no_relation, per_label_auprc_without_no_relation = safe_macro_auprc(positive_label_ids)
    micro_auprc = average_precision_score(y_true, all_probs, average="micro")
    return {
        "macro_auprc": macro_auprc,
        "micro_auprc": float(micro_auprc),
        "macro_auprc_without_no_relation": macro_auprc_without_no_relation,
        "per_label_auprc": per_label_auprc,
        "per_label_auprc_without_no_relation": per_label_auprc_without_no_relation,
    }


def evaluate(model, dataloader, device, id2label, no_relation_id=0, output_prediction_path=None):
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
            logits = model(**batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            pred_ids = preds.cpu().numpy().tolist()
            gold_ids = labels.cpu().numpy().tolist()
            prob_values = probs.cpu().numpy()
            all_preds.extend(pred_ids)
            all_labels.extend(gold_ids)
            all_probs.extend(prob_values)
            for guid, sentence, gold_label_text, pred_id, prob in zip(guids, sentences, label_texts, pred_ids, prob_values):
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
    micro_f1_without_no_relation = f1_score(filtered_labels, filtered_preds, average="micro") if filtered_labels else 0.0
    auprc_metrics = compute_auprc_metrics(all_labels, all_probs, id2label, no_relation_id=no_relation_id)
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
    if output_prediction_path is not None:
        os.makedirs(os.path.dirname(output_prediction_path), exist_ok=True)
        with open(output_prediction_path, "w", encoding="utf-8") as f:
            for row in prediction_rows:
                line = (
                    f"{row['guid']}\t{row['sentence']}\t{row['gold_label']}\t"
                    f"{row['pred_label']}\t{row['pred_score']:.6f}\n"
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
    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    train_dataset = REDataset(train_df, tokenizer, label2id, max_length=args.max_length)
    dev_dataset = REDataset(dev_df, tokenizer, label2id, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = REClassifier(args.model_name, num_labels=len(label2id), dropout=args.dropout)
    model.resize_token_embeddings(tokenizer)
    model.to(device)
    no_relation_id = label2id.get("no_relation", 0)
    if args.use_class_weight:
        class_weights = compute_class_weights(train_df, label2id).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -1.0
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            batch.pop("guid", None)
            batch.pop("sentence", None)
            batch.pop("label_text", None)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**batch)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": total_loss / step})
        avg_train_loss = total_loss / len(train_loader)
        prediction_path = os.path.join(args.output_dir, f"dev_predictions_epoch_{epoch}.tsv")
        metrics = evaluate(model, dev_loader, device, id2label, no_relation_id=no_relation_id, output_prediction_path=prediction_path)
        print(f"\nEpoch {epoch}")
        print(f"Train loss: {avg_train_loss:.4f}")
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
            torch.save({
                "mlp_state_dict": model.mlp.state_dict(),
                "bilinear_state_dict": model.bilinear.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "args": vars(args),
            }, os.path.join(save_dir, "model.pt"))
            best_prediction_path = os.path.join(args.output_dir, "best_dev_predictions.tsv")
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
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_dir)
    model = REClassifier(args.eval_model_dir, num_labels=len(label2id), dropout=args.dropout)
    model.resize_token_embeddings(tokenizer)
    model.mlp.load_state_dict(checkpoint["mlp_state_dict"])
    model.bilinear.load_state_dict(checkpoint["bilinear_state_dict"])
    model.to(device)
    model.eval()
    print("checkpoint scoring_type:", checkpoint.get("scoring_type"))
    print("checkpoint best_score:", checkpoint.get("best_score"))
    return model, tokenizer, label2id, id2label



def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model, tokenizer, label2id, id2label = load_trained_model(args, device)
    eval_df = read_tsv(args.eval_path)
    eval_dataset = REDataset(eval_df, tokenizer, label2id, max_length=args.max_length)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    no_relation_id = label2id.get("no_relation", 0)
    os.makedirs(args.output_dir, exist_ok=True)
    output_prediction_path = os.path.join(args.output_dir, args.eval_output_name)
    metrics = evaluate(model, eval_loader, device, id2label, no_relation_id=no_relation_id, output_prediction_path=output_prediction_path)
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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--train_path", type=str, default="refined-klue-re/train.tsv")
    parser.add_argument("--dev_path", type=str, default="refined-klue-re/dev.tsv")
    parser.add_argument("--eval_path", type=str, default="refined-klue-re/dev.tsv")
    parser.add_argument("--output_dir", type=str, default="outputs/re_klue_roberta_base_classifier_mlp_bilinear")
    parser.add_argument("--eval_model_dir", type=str, default="outputs/re_klue_roberta_base_classifier_mlp_bilinear/best_model")
    parser.add_argument("--eval_output_name", type=str, default="eval_predictions.tsv")
    parser.add_argument("--model_name", type=str, default="klue/roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
