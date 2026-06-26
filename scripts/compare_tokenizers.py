import argparse
import base64
import json


def load_hf(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = data["model"]
    return {
        "vocab": model["vocab"],
        "merges": [tuple(item) for item in model["merges"]],
        "special": {tok["content"]: tok["id"] for tok in data.get("added_tokens", [])},
    }


def load_go(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "model" in data and "vocab" in data["model"]:
        return load_hf(path)

    tokens = {
        base64.b64decode(value).decode("utf-8", errors="replace"): int(key)
        for key, value in data["tokens_base64"].items()
    }
    merges = []
    id_to_token = {
        int(key): base64.b64decode(value).decode("utf-8", errors="replace")
        for key, value in data["tokens_base64"].items()
    }
    for item in data["merges"]:
        a, b = item["pair"]
        merges.append((id_to_token[a], id_to_token[b]))
    return {"vocab": tokens, "merges": merges, "special": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", required=True)
    parser.add_argument("--go", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    hf = load_hf(args.hf)
    go = load_go(args.go)

    print(f"hf_vocab={len(hf['vocab'])} go_vocab={len(go['vocab'])}")
    print(f"hf_merges={len(hf['merges'])} go_merges={len(go['merges'])}")

    if hf["vocab"] == go["vocab"]:
        print("vocab: SAME")
    else:
        print("vocab: DIFFERENT")
        shown = 0
        for token, hf_id in hf["vocab"].items():
            go_id = go["vocab"].get(token)
            if go_id != hf_id:
                print(f"  token={token!r} hf={hf_id} go={go_id}")
                shown += 1
                if shown >= args.limit:
                    break

    if hf["merges"] == go["merges"]:
        print("merges: SAME")
    else:
        print("merges: DIFFERENT")
        for i, (h, g) in enumerate(zip(hf["merges"], go["merges"])):
            if h != g:
                print(f"  first_diff_merge={i} hf={h!r} go={g!r}")
                break


if __name__ == "__main__":
    main()
