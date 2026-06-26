import argparse
import json
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def iter_jsonl_texts(path, field, max_texts):
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = data.get(field)
            if not text:
                continue
            yield text
            count += 1
            if max_texts and count >= max_texts:
                break


def main():
    parser = argparse.ArgumentParser(description="Train a HuggingFace ByteLevel BPE tokenizer for parity checks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--field", default="text")
    parser.add_argument("--max-texts", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, required=True, help="Total vocab size, including special tokens.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--special-token", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=args.special_token,
    )

    start = time.perf_counter()
    tokenizer.train_from_iterator(
        iter_jsonl_texts(args.input, args.field, args.max_texts),
        trainer=trainer,
    )
    elapsed = time.perf_counter() - start

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"saved={tokenizer_path}")
    print(f"vocab_size={tokenizer.get_vocab_size()} train_time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
