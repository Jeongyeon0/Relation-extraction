import os
import json


def insert_entity_type_markers(sentence, subj, obj):
    entities = [
        {
            "start": subj["start_idx"],
            "end": subj["end_idx"],
            "open": f"[SUBJ-{subj['type']}] ",
            "close": f" [/SUBJ-{subj['type']}]"
        },
        {
            "start": obj["start_idx"],
            "end": obj["end_idx"],
            "open": f"[OBJ-{obj['type']}] ",
            "close": f" [/OBJ-{obj['type']}]"
        }
    ]

    # 뒤쪽부터 삽입해야 앞쪽 index가 밀리지 않음
    entities = sorted(entities, key=lambda x: x["start"], reverse=True)

    new_sentence = sentence

    for ent in entities:
        start = ent["start"]
        end = ent["end"] + 1  # KLUE의 end_idx는 inclusive라고 보는 것이 일반적

        new_sentence = (
            new_sentence[:start]
            + ent["open"]
            + new_sentence[start:end]
            + ent["close"]
            + new_sentence[end:]
        )

    return new_sentence


def extract_samples(data):

    # case 1: data 자체가 sample list인 경우
    if isinstance(data, list):
        return data

    # case 2: data 자체가 단일 sample인 경우
    if isinstance(data, dict) and "guid" in data:
        return [data]

    # case 3: dict 내부에 sample list가 들어 있는 경우
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict) and "guid" in value[0]:
                    return value

    raise ValueError(f"Cannot find valid samples. Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")


dir_path = "klue-re-v1.1/"
output_path = "refined-klue-re/"

os.makedirs(output_path, exist_ok=True)

for name in os.listdir(dir_path):
    if name.split('.')[0]=='relation_list':
        continue
    input_file_path = os.path.join(dir_path, name)

    if not os.path.isfile(input_file_path):
        continue

    # json 파일만 처리
    if not name.endswith(".json"):
        continue

    print(f"Processing: {name}")

    with open(input_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    samples = extract_samples(data)

    output_name = os.path.splitext(name)[0] + ".tsv"
    output_file_path = os.path.join(output_path, output_name)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for item in samples:
            guid = item["guid"]
            sentence = item["sentence"]
            subject_entity = item["subject_entity"]
            object_entity = item["object_entity"]
            label = item["label"]
            source = item["source"]

            marked_sentence = insert_entity_type_markers(
                sentence,
                subject_entity,
                object_entity
            )

            line = f"{guid}\t{marked_sentence}\t{label}\n"
            output_file.write(line)