# fast-bytelevel-bpe-go

[English](README.md) | 简体中文

一个用 Go 写的 HuggingFace 兼容 ByteLevel BPE tokenizer 训练器，面向大规模 JSONL 数据做了专门优化。在本地基准测试中，同样配置下约为 HuggingFace Python 训练路径的 6 倍速度。

`fast-bytelevel-bpe-go` 会输出 HuggingFace `tokenizers` 可直接加载的 `tokenizer.json`。它不是一个通用 tokenizer 框架，而是专注于常见的大 JSONL + ByteLevel BPE 训练场景，把性能路径做窄、做快。

一次本地 benchmark 中，它把同一个 32,779 词表大小的 tokenizer 从 4406.97s 缩短到 739.59s，并且 vocab 和 merges 与 HuggingFace 输出完全一致。

| 训练器 | 耗时 | 输出一致性 |
| --- | ---: | --- |
| HuggingFace Python binding | 4406.97s | baseline |
| fast-bytelevel-bpe-go | 739.59s | vocab 和 merges 完全一致 |

详细测试环境和说明见 [docs/benchmarks.md](docs/benchmarks.md)。

## 适用场景

当前项目刻意只支持一个窄而高性能的训练配置：

- JSONL 输入
- 单个字符串字段，默认 `text`
- `ByteLevel(add_prefix_space=false)`
- BPE model
- `initial_alphabet=ByteLevel.alphabet()`
- 与 HuggingFace 兼容的确定性 pair tie-break 逻辑

如果你的训练数据就是大规模 JSONL 文本，并且目标是生成 HuggingFace 可用的 ByteLevel BPE `tokenizer.json`，这个工具适合直接替换训练环节。

## 安装

```bash
go install github.com/meipian/fast-bytelevel-bpe-go/cmd/fastbpe@latest
```

本地开发构建：

```bash
go build -o bin/fastbpe ./cmd/fastbpe
```

## 使用

```bash
fastbpe \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 32768 \
  --out tokenizer.json
```

带 special tokens：

```bash
fastbpe \
  --input data.jsonl \
  --field text \
  --vocab-size 32779 \
  --special-token '<|bos|>' \
  --special-token '<|eos|>' \
  --special-token '<|pad|>' \
  --out tokenizer.json
```

`--vocab-size` 表示总词表大小，包含 special tokens。这个行为与 HuggingFace `BpeTrainer(vocab_size=...)` 一致。

## 与 HuggingFace 对齐校验

安装 Python 依赖：

```bash
python -m pip install tokenizers
```

训练 HuggingFace golden output：

```bash
python scripts/train_hf_tokenizer.py \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 512 \
  --output-dir /tmp/hf_tokenizer
```

使用 Go 版本训练：

```bash
go run ./cmd/fastbpe \
  --input examples/sample.jsonl \
  --field text \
  --vocab-size 512 \
  --out /tmp/go_tokenizer.json
```

比较输出：

```bash
python scripts/compare_tokenizers.py \
  --hf /tmp/hf_tokenizer/tokenizer.json \
  --go /tmp/go_tokenizer.json
```

期望结果：

```text
vocab: SAME
merges: SAME
```

## 为什么更快

这个训练器针对固定路径做了专门优化：

- 直接在 Go 中读取 JSONL
- 只提取配置的字符串字段，避免完整反序列化每条记录
- 只实现当前场景需要的 GPT-2/HF ByteLevel regex 行为
- 使用 packed `uint64` 存储 pair key
- 维护 `pair -> affected words` 索引
- 每次 merge 后只更新受影响的 words
- 对大批量 affected-word merge 更新做并行处理

这不代表 “Go 比 Rust 快”。HuggingFace `tokenizers` 是通用库，API 覆盖面更广；本项目的速度来自更窄的目标和更少的通用开销。

## 当前限制

目前不支持：

- 任意 normalizers
- 任意 pre-tokenizers
- `add_prefix_space=true`
- WordPiece 或 Unigram
- byte fallback
- continuing subword prefix / end of word suffix
- 非 JSONL 输入

输出文件是 HuggingFace 兼容的 `tokenizer.json`，可以这样加载：

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
```

## 许可证

MIT
