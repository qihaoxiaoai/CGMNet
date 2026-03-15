import argparse
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm


def _torch_load_compat(obj, map_location="cpu"):
    try:
        return torch.load(obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(obj, map_location=map_location)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return [_jsonable(v) for v in sorted(list(x), key=lambda z: str(z))]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _read_vocab(vocab_path: Path):
    with vocab_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        meta = {}
        try:
            meta = json.loads(header) if header else {}
        except Exception:
            meta = {}
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    frag_tokens = [ln.split("\t")[0] for ln in lines]
    vocab_size = len(frag_tokens) + 1
    return meta, frag_tokens, vocab_size


def _decode_frag_id(fid: int, frag_tokens: list[str]):
    if 0 <= int(fid) < len(frag_tokens):
        return frag_tokens[int(fid)]
    return f"<id={int(fid)}>"


def _ensure_repo_imports(repo_root: Path):
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cgmnet.utils.register import MODEL_REGISTRY, COLLATOR_REGISTRY
    import cgmnet.models.cgmnet
    import cgmnet.data.collator
    from cgmnet.data.featurizer import CGMNetFeaturizer
    from cgmnet.data.fragmentizer import load_vocab_meta
    from cgmnet.utils.fingerprint import get_batch_fingerprints

    return MODEL_REGISTRY, COLLATOR_REGISTRY, CGMNetFeaturizer, load_vocab_meta, get_batch_fingerprints


def _print_paths(args, out_dir, feature_dir, lmdb_dir, device):
    print(f"[repo_root]    {args.repo_root.resolve()}")
    print(f"[ckpt_path]    {args.ckpt_path.resolve()}")
    print(f"[vocab_path]   {args.vocab_path.resolve()}")
    print(f"[csv_path]     {args.csv_path.resolve()}")
    print(f"[out_dir]      {out_dir.resolve()}")
    print(f"[feature_dir]  {feature_dir.resolve()}")
    print(f"[graph_lmdb]   {lmdb_dir.resolve()}")
    if args.splits_dir is not None:
        print(f"[splits_dir]   {args.splits_dir.resolve()}")
        print(f"[scaffold_id]  {args.scaffold_id}")
    print(f"[device]       {device}")


def _detect_schema(df: pd.DataFrame):
    if "smiles" in df.columns:
        smiles_cols = ["smiles"]
    else:
        pat = re.compile(r"^smiles_(\d+)$")
        smiles_cols = sorted(
            [c for c in df.columns if pat.match(c)],
            key=lambda x: int(pat.match(x).group(1)),
        )
        if not smiles_cols:
            raise ValueError("CSV must contain 'smiles' or 'smiles_i' columns")

    ratio_pat = re.compile(r"^ratio_(\d+)$")
    ratio_cols_all = sorted(
        [c for c in df.columns if ratio_pat.match(c)],
        key=lambda x: int(ratio_pat.match(x).group(1)),
    )

    ratio_map = {}
    for sc in smiles_cols:
        m = re.match(r"^smiles_(\d+)$", sc)
        if m:
            rc = f"ratio_{m.group(1)}"
            ratio_map[sc] = rc if rc in df.columns else None
        else:
            ratio_map[sc] = None

    reserved = {"n_components", "smiles"} | set(smiles_cols) | set(ratio_cols_all)
    numeric_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        print("[WARN] no numeric label columns detected; y will be empty")

    return smiles_cols, ratio_map, numeric_cols


def _normalize_smiles_cell(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def _choose_rows(df_len: int, all_rows: bool, row_idx_list: list[int] | None):
    if all_rows or not row_idx_list:
        return np.arange(df_len, dtype=np.int64)
    seen = set()
    out = []
    for x in row_idx_list:
        i = int(x)
        if i < 0 or i >= df_len:
            raise IndexError(f"row_idx out of range: {i}")
        if i not in seen:
            seen.add(i)
            out.append(i)
    return np.array(out, dtype=np.int64)


def _collect_component_bank_and_mixtures(
    df: pd.DataFrame,
    rows_sel: np.ndarray,
    smiles_cols: list[str],
    ratio_map: dict[str, str | None],
    label_cols: list[str],
):
    comp_to_idx = {}
    comp_smiles = []

    K = len(smiles_cols)
    N = len(rows_sel)

    comp_idx = -np.ones((N, K), dtype=np.int64)
    ratios = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=np.bool_)
    n_components = np.zeros((N,), dtype=np.int64)

    y = np.zeros((N, len(label_cols)), dtype=np.float32) if label_cols else np.zeros((N, 0), dtype=np.float32)

    valid_row_mask = np.zeros((N,), dtype=np.bool_)

    for j, ridx in enumerate(rows_sel.tolist()):
        row = df.iloc[int(ridx)]

        row_smiles = []
        row_pos = []
        row_ratios_raw = []

        for pos, sc in enumerate(smiles_cols):
            smi = _normalize_smiles_cell(row[sc])
            if smi is None:
                continue
            row_smiles.append(smi)
            row_pos.append(pos)

            rc = ratio_map.get(sc, None)
            if rc is None or rc not in df.columns or pd.isna(row[rc]):
                row_ratios_raw.append(np.nan)
            else:
                try:
                    row_ratios_raw.append(float(row[rc]))
                except Exception:
                    row_ratios_raw.append(np.nan)

        if len(row_smiles) == 0:
            continue

        valid_row_mask[j] = True
        n_components[j] = len(row_smiles)

        rr = np.asarray(row_ratios_raw, dtype=np.float64)
        if np.isfinite(rr).all() and rr.sum() > 0:
            rr = rr / rr.sum()
        else:
            rr = np.full((len(row_smiles),), 1.0 / len(row_smiles), dtype=np.float64)

        for k_local, (smi, pos) in enumerate(zip(row_smiles, row_pos)):
            if smi not in comp_to_idx:
                comp_to_idx[smi] = len(comp_smiles)
                comp_smiles.append(smi)
            cid = comp_to_idx[smi]
            comp_idx[j, pos] = cid
            ratios[j, pos] = float(rr[k_local])
            mask[j, pos] = True

        if label_cols:
            vals = []
            for c in label_cols:
                v = row[c]
                vals.append(np.nan if pd.isna(v) else float(v))
            y[j] = np.asarray(vals, dtype=np.float32)

    return {
        "component_smiles": comp_smiles,
        "component_to_idx": comp_to_idx,
        "rows_sel": rows_sel,
        "comp_idx": comp_idx,
        "ratios": ratios,
        "mask": mask,
        "n_components": n_components,
        "y": y,
        "valid_row_mask": valid_row_mask,
    }


def _generate_features(smiles_list, feature_dir: Path, get_batch_fingerprints, n_jobs: int, features: list[str], rebuild: bool):
    feature_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    for name in features:
        fp_path = feature_dir / f"{name}.npy"
        if fp_path.exists() and (not rebuild):
            arr = np.load(fp_path, mmap_mode=None)
            if arr.shape[0] == len(smiles_list):
                print(f"[features] reuse {name}: {arr.shape}")
                out[name] = arr
                continue
            print(f"[features] shape mismatch for {name}, rebuilding: {arr.shape} vs {len(smiles_list)}")

        print(f"[features] generating {name} -> {fp_path}")
        arr = get_batch_fingerprints(smiles_list, name=name, n_jobs=n_jobs)
        arr = np.asarray(arr, dtype=np.float32)
        np.save(fp_path, arr)
        print(f"[features] saved {name}: {arr.shape}")
        out[name] = arr

    return out


def _build_featurizer(CGMNetFeaturizer, vocab_path: Path, vocab_meta: dict, args):
    frag_method = args.frag_method_override or vocab_meta.get("frag_method") or "relmole_vanilla"
    order = args.order_override
    if order is None:
        order = vocab_meta.get("line_order", 1)
    overlap = args.overlap_degree_override
    if overlap is None:
        overlap = vocab_meta.get("overlap_degree", None)

    print(f"[vocab_meta] {vocab_meta}")
    print(f"[frag_method]  {frag_method}")
    print(f"[line_order]   {order}")
    print(f"[overlap_deg]  {overlap}")

    featurizer = CGMNetFeaturizer(
        vocab_path=str(vocab_path),
        order=int(order),
        max_path_length=int(args.path_max_length),
        frag_method=str(frag_method),
        overlap_degree=(None if overlap is None else int(overlap)),
    )
    return featurizer, frag_method, int(order), (None if overlap is None else int(overlap))


def _featurize_one(featurizer, smiles: str):
    if hasattr(featurizer, "__call__"):
        out = featurizer(smiles)
    elif hasattr(featurizer, "featurize"):
        out = featurizer.featurize(smiles)
    else:
        raise RuntimeError("CGMNetFeaturizer has no callable/featurize interface")

    if isinstance(out, dict):
        g = out
    elif isinstance(out, (tuple, list)):
        g = out[0]
        if not isinstance(g, dict):
            raise RuntimeError("Featurizer returned tuple/list but first item is not dict")
    else:
        raise RuntimeError(f"Unsupported featurizer output type: {type(out)}")

    if "smiles" not in g:
        g["smiles"] = smiles
    return g


def _build_lmdb_for_components(featurizer, smiles_list: list[str], lmdb_dir: Path, rebuild: bool):
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = lmdb_dir / "meta.json"

    if (not rebuild) and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if int(meta.get("num_examples", -1)) == len(smiles_list) and int(meta.get("num_valid", -1)) == len(smiles_list):
                print(f"[lmdb] reuse {lmdb_dir}")
                return meta
        except Exception:
            pass

    print(f"[lmdb] building {lmdb_dir}")
    env = lmdb.open(
        str(lmdb_dir),
        map_size=8 * 1024**3,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    n_valid = 0
    with env.begin(write=True) as txn:
        for i, smi in enumerate(tqdm(smiles_list, desc="featurize->lmdb")):
            try:
                g = _featurize_one(featurizer, smi)
                buf = io.BytesIO()
                torch.save(g, buf)
                txn.put(str(i).encode("utf-8"), buf.getvalue())
                n_valid += 1
            except Exception as e:
                print(f"[WARN] featurize failed idx={i} smiles={smi} err={e}")

        txn.put(b"num_examples", str(len(smiles_list)).encode("utf-8"))
        txn.put(b"num_valid", str(n_valid).encode("utf-8"))

    env.sync()
    env.close()

    meta = {"num_examples": len(smiles_list), "num_valid": n_valid}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[lmdb] done | total={len(smiles_list)} valid={n_valid}")
    return meta


def _load_ckpt_flexible(path: Path):
    sd = _torch_load_compat(str(path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd)).startswith("module."):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


def _filter_state_dict_by_shape(model: torch.nn.Module, sd: dict):
    msd = model.state_dict()
    keep = {}
    dropped = 0
    for k, v in sd.items():
        if k not in msd:
            continue
        if hasattr(v, "shape") and hasattr(msd[k], "shape") and tuple(v.shape) != tuple(msd[k].shape):
            dropped += 1
            continue
        keep[k] = v
    return keep, dropped


def _build_model(MODEL_REGISTRY, vocab_size: int, vocab_path: Path, args, device):
    margs = SimpleNamespace(
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_mol_layers=int(args.n_mol_layers),
        n_subg_layers=int(args.n_subg_layers),
        in_feats=int(args.in_feats),
        feat_drop=float(args.feat_drop),
        attn_drop=float(args.attn_drop),
        ffn_drop=0.0,
        knodes=list(args.knodes),
        path_max_length=int(args.path_max_length),
        vocab_path=str(vocab_path),
    )
    ModelClass = MODEL_REGISTRY["cgmnet"]
    model = ModelClass(args=margs, n_tasks=vocab_size).to(device)

    sd = _load_ckpt_flexible(args.ckpt_path)
    sd_keep, dropped = _filter_state_dict_by_shape(model, sd)
    missing, unexpected = model.load_state_dict(sd_keep, strict=False)
    print(f"[ckpt] loaded={len(sd_keep)} missing={len(missing)} unexpected={len(unexpected)} dropped_shape={dropped}")

    model.eval()
    return model


def _move_to_device(batch: dict, device):
    for k, v in list(batch.items()):
        if isinstance(v, (torch.Tensor, dgl.DGLGraph)):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = {kk: (vv.to(device) if hasattr(vv, "to") else vv) for kk, vv in v.items()}
    return batch


def _open_lmdb_ro(lmdb_dir: Path):
    return lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=2048, subdir=True)


def _lmdb_load_graph(txn, idx: int):
    b = txn.get(str(int(idx)).encode("utf-8"))
    if b is None:
        return None
    return _torch_load_compat(io.BytesIO(b), map_location="cpu")


@torch.inference_mode()
def _export_component_embeddings(
    model,
    collator,
    lmdb_dir: Path,
    feature_bank: dict[str, np.ndarray],
    comp_smiles: list[str],
    frag_tokens: list[str],
    rep_dir: Path,
    device,
    batch_size: int,
):
    rep_dir.mkdir(parents=True, exist_ok=True)

    n = len(comp_smiles)
    d = int(model.d_model)

    mol_list = []
    frag_list = []
    frag_id_list = []
    frag_offsets = np.zeros((n + 1,), dtype=np.int64)

    component_rows = []
    frag_jsonl_path = rep_dir / "frag_info.jsonl"

    env = _open_lmdb_ro(lmdb_dir)
    n_missing = 0
    n_failed_batches = 0
    frag_cursor = 0

    with frag_jsonl_path.open("w", encoding="utf-8") as fw:
        for s in tqdm(range(0, n, batch_size), desc="export_all"):
            idxs = list(range(s, min(s + batch_size, n)))
            batch = []
            batch_comp_idx = []

            with env.begin(write=False) as txn:
                for i in idxs:
                    g = _lmdb_load_graph(txn, i)
                    if g is None:
                        n_missing += 1
                        continue
                    kn = {k: torch.from_numpy(np.asarray(v[i], dtype=np.float32)) for k, v in feature_bank.items()}
                    batch.append((g, torch.zeros(1, dtype=torch.float32), kn, comp_smiles[i]))
                    batch_comp_idx.append(i)

            if not batch:
                continue

            try:
                batched_data, _ = collator(batch)
                batched_data = _move_to_device(batched_data, device)
                mol_emb, frag_emb = model.encode(batched_data, return_frag=True)

                mol_np = mol_emb.detach().cpu().to(torch.float32).numpy()
                frag_np = frag_emb.detach().cpu().to(torch.float32).numpy()
                frag_g = batched_data["fragment_graph"]
                frag_ids = frag_g.ndata["id"].detach().cpu().numpy().astype(np.int64)
                frag_counts = frag_g.batch_num_nodes().detach().cpu().numpy().astype(np.int64)

                mol_list.append(mol_np)
                frag_list.append(frag_np)
                frag_id_list.append(frag_ids)

                ptr_local = np.zeros((len(frag_counts) + 1,), dtype=np.int64)
                ptr_local[1:] = np.cumsum(frag_counts)

                for bi, ci in enumerate(batch_comp_idx):
                    f0 = int(ptr_local[bi])
                    f1 = int(ptr_local[bi + 1])
                    local_ids = frag_ids[f0:f1].tolist()
                    local_smis = [_decode_frag_id(x, frag_tokens) for x in local_ids]

                    g0 = batch[bi][0]
                    final_details = g0.get("final_details", None)
                    group_atom_indices = g0.get("group_atom_indices", None)

                    component_rows.append(
                        {
                            "comp_idx": int(ci),
                            "smiles": str(comp_smiles[ci]),
                            "n_fragments": int(f1 - f0),
                        }
                    )

                    frag_info = {
                        "comp_idx": int(ci),
                        "smiles": str(comp_smiles[ci]),
                        "n_fragments": int(f1 - f0),
                        "frag_ids": local_ids,
                        "frag_smiles": local_smis,
                        "group_atom_indices": _jsonable(group_atom_indices) if group_atom_indices is not None else None,
                        "final_details": _jsonable(final_details) if final_details is not None else None,
                    }
                    fw.write(json.dumps(_jsonable(frag_info), ensure_ascii=False) + "\n")

                    frag_offsets[ci] = frag_cursor
                    frag_cursor += int(f1 - f0)

            except Exception as e:
                n_failed_batches += 1
                print(f"[WARN] batch failed [{s}:{s+batch_size}] err={e}")

    env.close()

    for i in range(n):
        if i == 0 and frag_offsets[i] != 0:
            frag_offsets[i] = 0
        if i > 0 and frag_offsets[i] == 0:
            frag_offsets[i] = frag_offsets[i - 1]
    frag_offsets[-1] = frag_cursor

    mol_embeddings = np.concatenate(mol_list, axis=0) if mol_list else np.zeros((0, d), dtype=np.float32)
    frag_embeddings = np.concatenate(frag_list, axis=0) if frag_list else np.zeros((0, d), dtype=np.float32)
    frag_token_ids = np.concatenate(frag_id_list, axis=0) if frag_id_list else np.zeros((0,), dtype=np.int64)

    if mol_embeddings.shape[0] != n:
        raise RuntimeError(f"mol_embeddings rows mismatch: {mol_embeddings.shape[0]} vs components={n}")
    if frag_offsets[-1] != frag_embeddings.shape[0]:
        raise RuntimeError(f"frag_offsets[-1]={frag_offsets[-1]} vs frag_embeddings={frag_embeddings.shape[0]}")
    if frag_token_ids.shape[0] != frag_embeddings.shape[0]:
        raise RuntimeError("frag_token_ids and frag_embeddings size mismatch")

    np.save(rep_dir / "mol_embeddings.npy", mol_embeddings)
    np.save(rep_dir / "frag_embeddings.npy", frag_embeddings)
    np.save(rep_dir / "frag_offsets.npy", frag_offsets)
    np.save(rep_dir / "frag_token_ids.npy", frag_token_ids)
    pd.DataFrame(component_rows).sort_values("comp_idx").to_csv(rep_dir / "component_rows.csv", index=False)

    summary = {
        "n_components": int(n),
        "n_missing_graph": int(n_missing),
        "n_failed_batches": int(n_failed_batches),
        "mol_embeddings_shape": list(mol_embeddings.shape),
        "frag_embeddings_shape": list(frag_embeddings.shape),
        "frag_offsets_shape": list(frag_offsets.shape),
    }
    (rep_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    print(f"[save] representation -> {rep_dir}")
    print(f"[summary] {summary}")

    return summary


def _load_scaffold_split(split_dir: Path, scaffold_id: int):
    p = split_dir / f"scaffold-{scaffold_id}.npy"
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")

    arr = np.load(p, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape != () and len(arr) == 3:
        return np.array(arr[0], dtype=np.int64), np.array(arr[1], dtype=np.int64), np.array(arr[2], dtype=np.int64)

    if isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
        d = arr.item()
        return np.array(d["train"], dtype=np.int64), np.array(d["valid"], dtype=np.int64), np.array(d["test"], dtype=np.int64)

    if isinstance(arr, (list, tuple)) and len(arr) == 3:
        return np.array(arr[0], dtype=np.int64), np.array(arr[1], dtype=np.int64), np.array(arr[2], dtype=np.int64)

    raise ValueError(f"unknown split format: {p}")


def _save_npz_pack(path: Path, pack: dict):
    np.savez_compressed(path, **pack)


def _export_mixture_pack(
    df: pd.DataFrame,
    out_dir: Path,
    schema_smiles_cols: list[str],
    schema_ratio_map: dict[str, str | None],
    label_cols: list[str],
    collect: dict,
    splits_dir: Path | None,
    scaffold_id: int,
):
    mix_dir = out_dir / "mixture"
    mix_dir.mkdir(parents=True, exist_ok=True)

    rows_sel = collect["rows_sel"]
    valid_row_mask = collect["valid_row_mask"]
    keep_idx = np.where(valid_row_mask)[0]

    rows_sel_kept = rows_sel[keep_idx]
    comp_idx = collect["comp_idx"][keep_idx]
    ratios = collect["ratios"][keep_idx]
    mask = collect["mask"][keep_idx]
    n_components = collect["n_components"][keep_idx]
    y = collect["y"][keep_idx] if collect["y"].size > 0 else np.zeros((len(keep_idx), 0), dtype=np.float32)

    row_df = df.iloc[rows_sel_kept].copy()
    row_df.insert(0, "row_idx", rows_sel_kept.astype(int))

    for j in range(comp_idx.shape[1]):
        row_df[f"__comp_idx_{j+1}"] = comp_idx[:, j]
        row_df[f"__ratio_used_{j+1}"] = ratios[:, j]
        row_df[f"__mask_{j+1}"] = mask[:, j].astype(int)

    row_df.to_csv(mix_dir / "mixtures.csv", index=False)

    smiles_cols = np.array(schema_smiles_cols, dtype=object)
    ratio_cols = np.array([schema_ratio_map[c] if schema_ratio_map[c] is not None else "" for c in schema_smiles_cols], dtype=object)
    label_cols_arr = np.array(label_cols, dtype=object)

    pack = {
        "row_idx": rows_sel_kept.astype(np.int64),
        "comp_idx": comp_idx.astype(np.int64),
        "ratios": ratios.astype(np.float32),
        "mask": mask.astype(np.bool_),
        "n_components": n_components.astype(np.int64),
        "y": y.astype(np.float32),
        "smiles_cols": smiles_cols,
        "ratio_cols": ratio_cols,
        "label_cols_numeric": label_cols_arr,
    }
    _save_npz_pack(mix_dir / "mixtures.npz", pack)

    split_summary = {}
    if splits_dir is not None:
        train_idx, valid_idx, test_idx = _load_scaffold_split(splits_dir, scaffold_id)
        split_root = mix_dir / f"splits_scaffold_{scaffold_id}"
        split_root.mkdir(parents=True, exist_ok=True)

        row_to_pos = {int(r): i for i, r in enumerate(rows_sel_kept.tolist())}

        for name, split_rows in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
            pos = [row_to_pos[int(r)] for r in split_rows.tolist() if int(r) in row_to_pos]
            pos = np.array(pos, dtype=np.int64)

            sp_pack = {
                "row_idx": rows_sel_kept[pos].astype(np.int64),
                "comp_idx": comp_idx[pos].astype(np.int64),
                "ratios": ratios[pos].astype(np.float32),
                "mask": mask[pos].astype(np.bool_),
                "n_components": n_components[pos].astype(np.int64),
                "y": y[pos].astype(np.float32),
                "smiles_cols": smiles_cols,
                "ratio_cols": ratio_cols,
                "label_cols_numeric": label_cols_arr,
            }
            _save_npz_pack(split_root / f"{name}.npz", sp_pack)
            split_summary[name] = int(len(pos))

    summary = {
        "n_rows_selected": int(len(rows_sel)),
        "n_rows_exported": int(len(rows_sel_kept)),
        "n_rows_skipped_no_component": int(len(rows_sel) - len(rows_sel_kept)),
        "K_max_components": int(comp_idx.shape[1]) if comp_idx.size else int(len(schema_smiles_cols)),
        "label_cols_numeric": [str(x) for x in label_cols],
        "split_counts": split_summary,
    }
    (mix_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    print(f"[save] mixtures -> {mix_dir}")
    print(f"[mixture_summary] {summary}")


def main():
    parser = argparse.ArgumentParser(description="Export CGMNet component/fragment representations for single- or multi-component CSVs.")
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)

    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--scaffold_id", type=int, default=0)

    parser.add_argument("--all_rows", action="store_true")
    parser.add_argument("--row_idx", type=int, nargs="*", default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--device_id", type=int, default=0)

    parser.add_argument("--rebuild_features", action="store_true")
    parser.add_argument("--rebuild_lmdb", action="store_true")

    parser.add_argument("--frag_method_override", type=str, default=None)
    parser.add_argument("--order_override", type=int, default=None)
    parser.add_argument("--overlap_degree_override", type=int, default=None)

    parser.add_argument("--knodes", type=str, nargs="*", default=["ecfp", "maccs", "torsion", "md"])

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_mol_layers", type=int, default=12)
    parser.add_argument("--n_subg_layers", type=int, default=2)
    parser.add_argument("--in_feats", type=int, default=137)
    parser.add_argument("--feat_drop", type=float, default=0.1)
    parser.add_argument("--attn_drop", type=float, default=0.1)
    parser.add_argument("--path_max_length", type=int, default=2)

    args = parser.parse_args()

    if args.repo_root is None:
        args.repo_root = Path(__file__).resolve().parents[1]

    out_dir = args.out_dir.resolve()
    feature_dir = out_dir / "features"
    lmdb_dir = out_dir / "lmdb"
    rep_dir = out_dir / "representation"

    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    _print_paths(args, out_dir, feature_dir, lmdb_dir, device)

    MODEL_REGISTRY, COLLATOR_REGISTRY, CGMNetFeaturizer, load_vocab_meta, get_batch_fingerprints = _ensure_repo_imports(args.repo_root)

    vocab_meta_raw, frag_tokens, vocab_size = _read_vocab(args.vocab_path)
    try:
        vm = load_vocab_meta(str(args.vocab_path))
        vocab_meta = {
            "frag_method": getattr(vm, "frag_method", None),
            "line_order": getattr(vm, "line_order", None),
            "overlap_degree": getattr(vm, "overlap_degree", None),
            "kekulize": getattr(vm, "kekulize", None),
        }
        for k, v in vocab_meta_raw.items():
            vocab_meta.setdefault(k, v)
    except Exception:
        vocab_meta = vocab_meta_raw

    df = pd.read_csv(args.csv_path)
    smiles_cols, ratio_map, label_cols = _detect_schema(df)

    rows_sel = _choose_rows(len(df), args.all_rows, args.row_idx)
    collect = _collect_component_bank_and_mixtures(df, rows_sel, smiles_cols, ratio_map, label_cols)

    comp_smiles = collect["component_smiles"]
    if len(comp_smiles) == 0:
        raise RuntimeError("No valid component SMILES found in selected rows")

    pd.DataFrame(
        {
            "comp_idx": np.arange(len(comp_smiles), dtype=np.int64),
            "smiles": comp_smiles,
        }
    ).to_csv(out_dir / "component_bank.csv", index=False)

    feature_bank = _generate_features(
        smiles_list=comp_smiles,
        feature_dir=feature_dir,
        get_batch_fingerprints=get_batch_fingerprints,
        n_jobs=int(args.n_jobs),
        features=list(args.knodes),
        rebuild=bool(args.rebuild_features),
    )

    featurizer, frag_method, line_order, overlap_degree = _build_featurizer(
        CGMNetFeaturizer=CGMNetFeaturizer,
        vocab_path=args.vocab_path,
        vocab_meta=vocab_meta,
        args=args,
    )

    _build_lmdb_for_components(
        featurizer=featurizer,
        smiles_list=comp_smiles,
        lmdb_dir=lmdb_dir,
        rebuild=bool(args.rebuild_lmdb),
    )

    CollatorClass = COLLATOR_REGISTRY["cgmnet_finetune"]
    collator = CollatorClass()

    model = _build_model(
        MODEL_REGISTRY=MODEL_REGISTRY,
        vocab_size=vocab_size,
        vocab_path=args.vocab_path,
        args=args,
        device=device,
    )

    rep_summary = _export_component_embeddings(
        model=model,
        collator=collator,
        lmdb_dir=lmdb_dir,
        feature_bank=feature_bank,
        comp_smiles=comp_smiles,
        frag_tokens=frag_tokens,
        rep_dir=rep_dir,
        device=device,
        batch_size=int(args.batch_size),
    )

    _export_mixture_pack(
        df=df,
        out_dir=out_dir,
        schema_smiles_cols=smiles_cols,
        schema_ratio_map=ratio_map,
        label_cols=label_cols,
        collect=collect,
        splits_dir=args.splits_dir,
        scaffold_id=int(args.scaffold_id),
    )

    top_summary = {
        "csv_path": str(args.csv_path.resolve()),
        "n_csv_rows": int(len(df)),
        "schema": {
            "smiles_cols": smiles_cols,
            "ratio_cols": {k: v for k, v in ratio_map.items()},
            "label_cols_numeric": label_cols,
        },
        "vocab_meta": vocab_meta,
        "featurizer": {
            "frag_method": frag_method,
            "line_order": line_order,
            "overlap_degree": overlap_degree,
            "path_max_length": int(args.path_max_length),
        },
        "representation_summary": rep_summary,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(_jsonable(top_summary), ensure_ascii=False, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
