#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter a fragment vocabulary by fragment size and frequency.

用法示例：
    python 02_filter_vocab.py \
        --input_vocab ../jobs/big/vocabs/vocab.txt \
        --output_vocab ../jobs/big/vocabs/vocab_filtered.txt \
        --max_atoms 25 \
        --min_count 5

假设 vocab.txt 格式为：
    第一行：JSON 头部，记录 kekulize / frag_method / line_order 等信息
    之后每行：<frag_smi> <n_heavy_atoms> <count>

脚本会：
  - 丢弃 n_heavy_atoms > max_atoms 的碎片
  - 丢弃 count < min_count 的碎片
  - 单原子碎片（n_heavy_atoms == 1）可选择强制保留（默认保留）
  - header 中追加一些统计字段，方便记录过滤信息
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_vocab_line(line: str) -> Tuple[str, int, int] | None:
    """
    解析一行 vocab（非 JSON header）：
        <frag_smi> <n_heavy_atoms> <count>

    返回 (smi, n_heavy, count)，解析失败则返回 None。
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 3:
        # 兼容：如果将来你有别的格式，可以在这里扩展逻辑
        print(f"[WARN] Cannot parse vocab line (need 3 cols): {line}")
        return None

    smi = parts[0]
    try:
        n_heavy = int(parts[1])
        count = int(parts[2])
    except ValueError:
        print(f"[WARN] Cannot cast to int in vocab line: {line}")
        return None

    return smi, n_heavy, count


def filter_vocab(
    input_path: Path,
    output_path: Path,
    max_atoms: int,
    min_count: int,
    keep_single_atom: bool = True,
):
    """
    读取 input_path 的 vocab，按 max_atoms / min_count 过滤，
    写入 output_path。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input vocab not found: {input_path}")

    print(f"=== Loading vocab from: {input_path} ===")

    with input_path.open("r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Input vocab file is empty: {input_path}")

    # 1) 读取 JSON header
    header_line = lines[0].strip()
    try:
        header = json.loads(header_line) if header_line else {}
    except json.JSONDecodeError:
        print("[WARN] First line is not valid JSON, treat as empty header.")
        header = {}

    # 2) 解析碎片行
    entries: List[Tuple[str, int, int]] = []
    for raw in lines[1:]:
        parsed = parse_vocab_line(raw)
        if parsed is None:
            continue
        entries.append(parsed)

    total_fragments = len(entries)
    total_count = sum(c for _, _, c in entries)

    print(f"Total fragments (distinct): {total_fragments}")
    print(f"Total occurrences        : {total_count}")

    # 3) 过滤
    kept_entries: List[Tuple[str, int, int]] = []
    removed_by_size = 0
    removed_by_freq = 0

    for smi, n_heavy, count in entries:
        # 单原子碎片可选择强制保留
        if keep_single_atom and n_heavy == 1:
            kept_entries.append((smi, n_heavy, count))
            continue

        if n_heavy > max_atoms:
            removed_by_size += 1
            continue
        if count < min_count:
            removed_by_freq += 1
            continue

        kept_entries.append((smi, n_heavy, count))

    kept_fragments = len(kept_entries)
    kept_count = sum(c for _, _, c in kept_entries)
    coverage = kept_count / total_count if total_count > 0 else 0.0

    print("\n=== Filtering summary ===")
    print(f"  max_atoms       = {max_atoms}")
    print(f"  min_count       = {min_count}")
    print(f"  keep_single_atom= {keep_single_atom}")
    print(f"  kept fragments  = {kept_fragments} / {total_fragments}")
    print(f"  removed_by_size = {removed_by_size}")
    print(f"  removed_by_freq = {removed_by_freq}")
    print(f"  kept occurrences= {kept_count} / {total_count} "
          f"({coverage*100:.4f}% coverage)")

    # 4) 更新 header 信息（不破坏原字段）
    header = dict(header)  # 拷贝一份
    header["filtered"] = True
    header["filter_max_atoms"] = max_atoms
    header["filter_min_count"] = min_count
    header["filter_keep_single_atom"] = keep_single_atom
    header["orig_num_fragments"] = total_fragments
    header["orig_total_count"] = total_count
    header["kept_num_fragments"] = kept_fragments
    header["kept_total_count"] = kept_count
    header["kept_coverage"] = coverage

    # 5) 写出新的 vocab
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for smi, n_heavy, count in kept_entries:
            f.write(f"{smi}\t{n_heavy}\t{count}\n")

    print(f"\nFiltered vocab written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter fragment vocabulary by size & frequency.",
    )
    parser.add_argument(
        "--input_vocab",
        type=Path,
        required=True,
        help="Path to the original vocab.txt",
    )
    parser.add_argument(
        "--output_vocab",
        type=Path,
        default=None,
        help="Path to save filtered vocab. "
             "If not set, will use <input_dir>/vocab_filtered.txt",
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=25,
        help="Maximum heavy-atom count for a fragment to be kept.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum global frequency for a fragment to be kept.",
    )
    parser.add_argument(
        "--no_keep_single_atom",
        action="store_true",
        help="By default, single-atom fragments (n_heavy==1) are always kept. "
             "Set this flag to disable that behavior.",
    )

    args = parser.parse_args()

    if args.output_vocab is None:
        args.output_vocab = args.input_vocab.with_name("vocab_filtered.txt")

    filter_vocab(
        input_path=args.input_vocab,
        output_path=args.output_vocab,
        max_atoms=args.max_atoms,
        min_count=args.min_count,
        keep_single_atom=not args.no_keep_single_atom,
    )


if __name__ == "__main__":
    main()

