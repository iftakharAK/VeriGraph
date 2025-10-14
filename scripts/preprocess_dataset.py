
import json
import argparse
from tqdm import tqdm


def preprocess(input_path, output_path):
    processed = []
    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Preprocessing"):
            data = json.loads(line)
            if "statements" not in data:
                continue
            sample = {
                "id": data.get("id"),
                "statements": [s.strip() for s in data["statements"]],
                "labels": data.get("labels", []),
            }
            processed.append(sample)

    with open(output_path, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    print(f"Preprocessed {len(processed)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw dataset.jsonl")
    parser.add_argument("--output", required=True, help="Path to save processed file")
    args = parser.parse_args()

    preprocess(args.input, args.output)