"""PEFT pattern paths and normalized writer targets must not lose serialized edits."""

import json
import re

import numpy as np
import pytest

from adapter_fakes import CFG, factors, write_adapter, write_safetensors
from tensorfold.families.deepseek_v41.cuda import lora_format as lf

DIMS = lf.Dims.of(CFG)
KINDS = ("attn.wo_b", "ffn.shared_experts.w2")
PREFIXES = ("", "model.", "base_model.model.", "base_model.model.model.")


def pair(kind="attn.wo_b", rank=1, zero=False):
    width = DIMS.o_in if kind == "attn.wo_b" else DIMS.inter
    a, b = factors(np.random.default_rng(47), DIMS.dim, width, rank)
    return (a * 0, b * 0) if zero else (a, b)


def test_wrapped_model_alpha_pattern(tmp_path):
    folder = write_adapter(tmp_path / "a", {"layers.1.attn.wo_b": pair()},
        prefix="base_model.model.model.", config={"r": 1, "lora_alpha": 1,
        "alpha_pattern": {r"^model\.layers\.1\.attn\.wo_b": 4}})
    (module,) = lf.inspect_folder(folder, DIMS).modules
    assert (module.rank, module.alpha, module.scale) == (1, 4.0, 4.0)


def test_duplicate_normalized_target(tmp_path):
    folder = write_adapter(tmp_path / "a", {"layers.1.attn.wo_b": pair(), "layers.01.attn.wo_b": pair()}, prefix="")
    with pytest.raises(lf.AdapterError, match="duplicate normalized target"):
        lf.inspect_folder(folder, DIMS)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("field", ["rank", "alpha"])
@pytest.mark.parametrize("matches", [False, True], ids=["other-anchor", "own-anchor"])
def test_anchored_patterns_match_only_the_original_inner_path(tmp_path, prefix, field, matches):
    inner = "model." if prefix in ("model.", "base_model.model.model.") else ""
    anchor = inner if matches else ("" if inner else "model.")
    pattern = "^" + re.escape(anchor + "layers.1.attn.wo_b")
    rank = 2 if field == "rank" and matches else 1
    config = {"r": 1, "lora_alpha": 1, field + "_pattern": {pattern: 2 if field == "rank" else 4}}
    folder = write_adapter(tmp_path / "a", {"layers.1.attn.wo_b": pair(rank=rank)}, prefix=prefix, config=config)
    (module,) = lf.inspect_folder(folder, DIMS).modules
    alpha = 4.0 if field == "alpha" and matches else 1.0
    assert (module.name, module.rank, module.alpha, module.scale) == ("layers.1.attn.wo_b", rank, alpha, alpha / rank)


@pytest.mark.parametrize("field", ["rank", "alpha"])
def test_a_pattern_cannot_restore_the_removed_peft_wrapper(tmp_path, field):
    pattern = r"^base_model\.model\.model\.layers\.1\.attn\.wo_b"
    folder = write_adapter(tmp_path / "a", {"layers.1.attn.wo_b": pair()}, prefix="base_model.model.model.",
        config={"r": 1, "lora_alpha": 1, field + "_pattern": {pattern: 2 if field == "rank" else 4}})
    (module,) = lf.inspect_folder(folder, DIMS).modules
    assert (module.rank, module.alpha, module.scale) == (1, 1.0, 1.0)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("kind", KINDS)
def test_rank_and_alpha_suffix_patterns_keep_independent_first_match_order(tmp_path, prefix, kind):
    name = "layers.1." + kind
    suffix = re.escape(kind)
    folder = write_adapter(tmp_path / "a", {name: pair(kind, rank=2)}, prefix=prefix, config={
        "r": 1, "lora_alpha": 1, "rank_pattern": {suffix: 2, re.escape(name): 3},
        "alpha_pattern": {suffix: 8, re.escape(name): 12}})
    (module,) = lf.inspect_folder(folder, DIMS).modules
    assert (module.rank, module.alpha, module.scale) == (2, 8.0, 4.0)


ALIASES = (("layers.1.", "layers.01."), ("layers.1.", "model.layers.1."),
           ("base_model.model.layers.1.", "base_model.model.model.layers.01."),
           ("base_model.model.model.layers.1.", "model.layers.01."))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("aliases", ALIASES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rank,zero", [(1, False), (2, True)], ids=["nonzero-rank1", "zero-rank2"])
def test_duplicate_writers_are_refused_for_each_alias_and_file_order(tmp_path, kind, aliases, reverse, rank, zero):
    names = [prefix + kind for prefix in aliases]
    if reverse:
        names.reverse()
    folder = write_adapter(tmp_path / "a", {name: pair(kind, rank, zero) for name in names}, prefix="")
    with pytest.raises(lf.AdapterError, match="duplicate normalized target"):
        lf.inspect_folder(folder, DIMS)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_halves_from_distinct_original_pattern_paths_are_not_combined(tmp_path, kind, reverse):
    a, b = pair(kind)
    folder = write_adapter(tmp_path / "a", {"layers.1." + kind: (a, b)})
    paths = ["base_model.model.layers.1.", "base_model.model.model.layers.1."]
    if reverse:
        paths.reverse()
    write_safetensors(folder / lf.WEIGHTS, {
        paths[0] + kind + ".lora_A.weight": ("F32", list(a.shape), a.tobytes()),
        paths[1] + kind + ".lora_B.weight": ("F32", list(b.shape), b.tobytes())})
    with pytest.raises(lf.AdapterError, match=r"lora_[AB] is missing"):
        lf.inspect_folder(folder, DIMS)


def test_single_numeric_alias_and_distinct_writer_targets_remain_valid(tmp_path):
    modules = {"layers.01.attn.wo_b": pair(), "layers.1.ffn.shared_experts.w2": pair(KINDS[1]),
               "layers.2.attn.wo_b": pair()}
    spec = lf.inspect_folder(write_adapter(tmp_path / "a", modules, prefix="base_model.model.model."), DIMS)
    assert {(m.layer, m.kind) for m in spec.modules} == {(1, "wo_b"), (1, "shared_w2"), (2, "wo_b")}
    assert len(spec.describe()["modules"]) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_a_collapsed_optional_manifest_cannot_hide_a_duplicate_writer(tmp_path, kind):
    folder = write_adapter(tmp_path / "a", {"layers.1." + kind: pair(kind), "layers.01." + kind: pair(kind)}, prefix="")
    (folder / lf.MANIFEST).write_text(json.dumps({"format": lf.FORMAT,
        "files": {lf.WEIGHTS: {"sha256": lf.sha256_file(folder / lf.WEIGHTS)}},
        "modules": {"layers.1." + kind: {"rank": 1, "alpha": 1, "scale": 1}}}))
    with pytest.raises(lf.AdapterError, match="duplicate normalized target"):
        lf.inspect_folder(folder, DIMS)


@pytest.mark.parametrize("prefix", ["base_model.model.base_model.model.", "model.model."])
def test_repeated_wrappers_are_not_new_supported_writer_paths(tmp_path, prefix):
    folder = write_adapter(tmp_path / "a", {"layers.1.attn.wo_b": pair()}, prefix=prefix)
    with pytest.raises(lf.AdapterError, match="not a module this engine can edit"):
        lf.inspect_folder(folder, DIMS)
