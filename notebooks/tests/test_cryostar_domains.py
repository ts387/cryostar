"""Tests for notebooks/cryostar_domains.py (DOMAINS / DOMAIN_FILE in the Colab notebook's 3.1).

Run from the repository root:  python -m pytest notebooks/tests -q
The acceptance test on a real model runs when CRYOSTAR_DOMAIN_TEST_PDB names that model's .pdb file.
"""
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1]
REPO = NOTEBOOKS.parent
sys.path[:0] = [str(NOTEBOOKS), str(REPO)]

import cryostar_domains as D  # noqa: E402

EXAMPLE_CFG = REPO / "projects" / "star" / "atom_configs" / "example.py"
USER_SPEC = "L27=C:1-107; PDZ=C:108-214; SH3GUK=C:215-284,C:338-540; peptide=C:293-306; flex=C:285-292,C:307-337"
LOSS = dict(intra_chain_cutoff=12.0, inter_chain_cutoff=0.0, intra_chain_res_bound=None, nt_intra_chain_cutoff=15.0,
            nt_inter_chain_cutoff=15.0, nt_intra_chain_res_bound=None, sse_weight=0.0)


# ---------- a small synthetic model ----------
def _pdb_line(i, atom, res_name, chain, res_id, xyz):
    name = f" {atom:<3}" if len(atom) < 4 else atom
    return (f"ATOM  {i:5d} {name} {res_name:>3} {chain}{res_id:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C")


def _helix(n, start, axis_offset):
    t = np.arange(n)
    return np.stack([2.3 * np.cos(t * 1.75) + axis_offset[0], 2.3 * np.sin(t * 1.75) + axis_offset[1],
                     1.5 * t + start], axis=1)


@pytest.fixture(scope="module")
def model_pdb(tmp_path_factory):
    """Chain A: residues 1-15 and 18-30 (a gap); chain B: 1-20 next to A, so some A-B beads are within 12 Å."""
    lines, i = [], 1
    res_a = list(range(1, 16)) + list(range(18, 31))
    for res, xyz in zip(res_a, _helix(len(res_a), 0.0, (0.0, 0.0))):
        lines.append(_pdb_line(i, "CA", "ALA", "A", res, xyz))
        i += 1
    for res, xyz in zip(range(1, 21), _helix(20, 5.0, (9.0, 0.0))):
        lines.append(_pdb_line(i, "CA", "GLY", "B", res, xyz))
        i += 1
    path = tmp_path_factory.mktemp("model") / "model.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


@pytest.fixture(scope="module")
def meta(model_pdb):
    from cryostar.utils.polymer import Polymer
    return Polymer.from_pdb(str(model_pdb))


def _direct_pairs(meta, labels, cutoff=12.0, inter=0.0):
    """Restraint pairs computed without CryoSTAR's functions: i < j within `cutoff` in the same group (or within
    `inter` in different groups), minus sequence neighbours (consecutive residue numbers in one chain)."""
    out = set()
    n = len(meta.chain_id)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(meta.coord[i] - meta.coord[j])
            if not (d <= cutoff if labels[i] == labels[j] else d <= inter):
                continue
            if meta.chain_id[i] == meta.chain_id[j] and abs(int(meta.res_id[i]) - int(meta.res_id[j])) == 1:
                continue
            out.add((i, j))
    return out


def _as_set(pairs):
    return {tuple(map(int, p)) for p in pairs}


# ---------- parsing ----------
def test_parse_basic_and_whitespace():
    entries = D.domain_parse("  L27 = C:1-107 ;SH3GUK=C:215 - 284 , C:338–540; lig=C:300;flex=C:285-292; flex=C:310;")
    assert [(e["name"], e["flex"], e["segments"]) for e in entries] == [
        ("L27", False, [("C", 1, 107)]),
        ("SH3GUK", False, [("C", 215, 284), ("C", 338, 540)]),
        ("lig", False, [("C", 300, 300)]),
        ("flex", True, [("C", 285, 292)]),
        ("flex", True, [("C", 310, 310)])]


def test_parse_blank_negative_and_flex_case():
    assert D.domain_parse("") == [] and D.domain_parse("   ") == []
    e = D.domain_parse("N=A:-5--1,A:0; FLEX=A:3-4")
    assert e[0]["segments"] == [("A", -5, -1), ("A", 0, 0)] and e[1] == dict(name="flex", flex=True,
                                                                              segments=[("A", 3, 4)])


@pytest.mark.parametrize("text, needle", [
    ("L27 C:1-107", "has no '='"),
    ("=C:1-10", "has no name"),
    ("my domain=C:1-10", "not a usable name"),
    ("L27=1-107", "Give the chain too, e.g. C:1-107"),
    ("L27=C1-107", "not a residue range"),
    ("L27=C:1-", "not a residue range"),
    ("L27=C:a-b", "not a residue range"),
    ("L27=", "lists no residues"),
    ("L27=C:107-1", "starts after it ends"),
    ("A=C:1-10; a=C:20-30", "The name a is used twice (A and a)."),
    ("N=C:1-10; N=C:20-30", "The name N is used twice. Give each domain one entry"),
    ("A=C:1-10; B=C:10-20", "A (C:1-10) and B (C:10-20) overlap at C:10"),
    ("A=C:1-100; B=C:20-30; C=C:40-50", "overlap"),
    ("A=C:1-10,C:5-20", "C:1-10 and C:5-20 in A overlap at C:5-10"),
    ("A=C:1-10; flex=C:8-12", "overlap"),
    ("flex=C:1-10; flex=C:10", "overlap"),
    (";;", "lists no domains"),
])
def test_parse_errors(text, needle):
    with pytest.raises(D.DomainSpecError, match=None) as err:
        D.domain_parse(text)
    assert needle in str(err.value), str(err.value)


def test_same_range_on_two_chains_is_not_an_overlap():
    assert len(D.domain_parse("a=A:1-10; b=B:1-10")) == 2


# ---------- labels ----------
CHAIN = np.array(list("AAAAAAAAAA") + list("BBBBB"))
RES = np.array(list(range(1, 11)) + [1, 2, 3, 5, 6])


def test_labels_unlisted_flex_and_order():
    out = D.domain_labels(D.domain_parse("d1=A:3-5; flex=A:6-7; d2=A:8-10,B:1-2"), CHAIN, RES)
    # A1-2 rest of A | d1 | 2 flex singletons | d2 (also on B) | rest of B
    assert out["labels"].tolist() == [0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5]
    assert out["labels"].dtype == np.int64
    assert [(g["name"], g["kind"], g["labels"]) for g in out["groups"]] == [
        ("d1", "domain", [1]), ("flex", "flex", [2, 3]), ("d2", "domain", [4]),
        ("rest of chain A", "chain", [0]), ("rest of chain B", "chain", [5])]
    assert out["warnings"] == []


def test_labels_whole_chain_untouched_is_named_chain():
    out = D.domain_labels(D.domain_parse("d=A:1-10"), CHAIN, RES)
    assert [g["name"] for g in out["groups"]] == ["d", "chain B"]
    assert out["labels"].tolist() == [0] * 10 + [1] * 5


def test_labels_errors_and_warnings():
    with pytest.raises(D.DomainSpecError, match="has no chain C .*A, B"):
        D.domain_labels(D.domain_parse("d=C:1-5"), CHAIN, RES)
    with pytest.raises(D.DomainSpecError, match="None of the residues of d .B:4. is in the model"):
        D.domain_labels(D.domain_parse("d=B:4"), CHAIN, RES)
    out = D.domain_labels(D.domain_parse("d=B:2-8"), CHAIN, RES)
    assert out["warnings"] == ["d: 3 residue(s) of B:2-8 are not in the model (B:4, B:7-8)."]


def test_labels_from_file(tmp_path):
    good = tmp_path / "good.npy"
    np.save(good, np.array([5, 5, 7], dtype=np.int32))
    out = D.domain_labels_from_file(f"  {good} ", 3)
    assert out["labels"].dtype == np.int64 and out["labels"].tolist() == [5, 5, 7]
    assert [g["name"] for g in out["groups"]] == ["label 5", "label 7"]
    cases = {"missing.npy": "does not exist", "short.npy": "has 2 labels, but the model has 3 beads",
             "float.npy": "float64 array", "twod.npy": "shape (3, 1)", "obj.npy": "without pickle",
             "arch.npz": ".npz archive", "text.npy": "without pickle"}
    np.save(tmp_path / "short.npy", np.array([1, 2]))
    np.save(tmp_path / "float.npy", np.array([1.0, 2.0, 3.0]))
    np.save(tmp_path / "twod.npy", np.zeros((3, 1), dtype=int))
    np.save(tmp_path / "obj.npy", np.array([1, "a", None], dtype=object), allow_pickle=True)
    np.savez(tmp_path / "arch.npz", a=np.arange(3))
    (tmp_path / "text.npy").write_text("hello")
    for name, needle in cases.items():
        with pytest.raises(D.DomainSpecError) as err:
            D.domain_labels_from_file(tmp_path / name, 3)
        assert needle in str(err.value), (name, str(err.value))


def test_same_grouping():
    assert D.domain_same_grouping([3, 3, 1, 2], [0, 0, 5, 9])
    assert not D.domain_same_grouping([3, 3, 1, 2], [0, 0, 5, 5])
    assert not D.domain_same_grouping([1, 1], [1, 1, 1])


def test_save_is_content_addressed(tmp_path):
    p1 = D.domain_save(np.array([0, 0, 1], dtype=np.int32), tmp_path / "prep")
    assert p1.name.startswith("domains_") and np.load(p1).dtype == np.int64
    mtime = p1.stat().st_mtime_ns
    assert D.domain_save([0, 0, 1], tmp_path / "prep") == p1 and p1.stat().st_mtime_ns == mtime
    assert D.domain_save([0, 1, 1], tmp_path / "prep") != p1


# ---------- pair counting (mirrors train_atom.py) ----------
def test_chain_labels_equal_chain_baseline(meta):
    connect, by_chain = D.domain_restraint_pairs(meta, LOSS)
    as_int = np.unique(meta.chain_id, return_inverse=True)[1].reshape(-1)
    _, by_label = D.domain_restraint_pairs(meta, LOSS, as_int)
    assert _as_set(by_chain) == _as_set(by_label) and len(by_chain) > 0
    assert _as_set(by_chain) == _direct_pairs(meta, as_int)
    # continuity: consecutive residue numbers in each chain (A has a gap between 15 and 18)
    assert len(connect) == (14 + 12) + 19


def test_domain_counts_vs_direct(meta):
    plan = D.domain_labels(D.domain_parse("n=A:1-12; flex=A:13-20; c=A:21-30,B:12-20"), meta.chain_id, meta.res_id)
    _, pairs = D.domain_restraint_pairs(meta, LOSS, plan["labels"])
    direct = _direct_pairs(meta, plan["labels"])
    assert _as_set(pairs) == direct
    counts = D.domain_pair_counts(pairs, meta.chain_id, plan["labels"])
    cross = {(i, j) for i, j in direct if meta.chain_id[i] != meta.chain_id[j]}
    assert counts["cross_chain"] == len(cross) > 0  # domain c spans chains A and B
    assert counts["total"] == len(direct) and counts["between_labels"] == 0
    assert sum(counts["by_chain"].values()) + counts["cross_chain"] == counts["total"]
    flex = plan["groups"][1]["labels"]
    assert all(counts["by_label"].get(v, 0) == 0 for v in flex) and len(flex) == 6  # A:13-15 and A:18-20


def test_flex_inter_cutoff(meta):
    """With inter_chain_cutoff > 0, train_atom.py also restrains beads of different labels that are close enough."""
    labels = np.arange(len(meta.chain_id))  # everything flexible
    _, pairs = D.domain_restraint_pairs(meta, dict(LOSS, inter_chain_cutoff=6.0), labels)
    assert _as_set(pairs) == _direct_pairs(meta, labels, inter=6.0) != set()
    _, none = D.domain_restraint_pairs(meta, LOSS, labels)
    assert len(none) == 0


def test_empty_pairs_would_break_train_atom(meta):
    """CryoSTAR's remove_duplicate_pairs fails on an empty array, so domain_plan refuses labels that leave no pair."""
    from cryostar.utils.dist_loss import remove_duplicate_pairs
    connect, _ = D.domain_restraint_pairs(meta, LOSS)
    with pytest.raises(ValueError):
        remove_duplicate_pairs(np.zeros((0, 2), dtype=np.int64), connect)
    with pytest.raises(D.DomainSpecError, match="no restraint pair is left"):
        D.domain_plan(meta, LOSS, "flex=A:1-15,A:18-30,B:1-20")


def test_only_cross_chain_pairs_refused():
    m = SimpleNamespace(chain_id=np.array(["A", "A", "B"]), res_id=np.array([1, 2, 1]),
                        atom_name=np.array(["CA"] * 3), coord=np.array([[0., 0, 0], [3.8, 0, 0], [0, 5, 0]]))
    with pytest.raises(D.DomainSpecError, match="joins two different chains"):
        D.domain_plan(m, LOSS, "x=A:1,B:1; flex=A:2")


def test_sse_pairs_removed(meta):
    loss = dict(LOSS, sse_weight=1.0)
    _, pairs = D.domain_restraint_pairs(meta, loss)
    from cryostar.utils.dist_loss import find_quaint_cutoff_pairs
    sse = _as_set(find_quaint_cutoff_pairs(meta.coord, meta.chain_id, meta.res_id, 12.0, 0, 20))
    assert _as_set(pairs) == _direct_pairs(meta, meta.chain_id) - sse


def test_nucleotide_branch(tmp_path):
    """Grouped by chain, train_atom.py uses the nt_* cutoffs for nucleotides; with labels it uses the protein ones."""
    from cryostar.utils.polymer import Polymer
    lines = [_pdb_line(i + 1, "CA", "ALA", "A", i + 1, (3.8 * i, 0.0, 0.0)) for i in range(6)]
    lines += [_pdb_line(10 + i, "C1'", base, "R", i + 1, (7.0 * i, 8.0, 0.0)) for i, base in enumerate("AUGCA")]
    lines += [_pdb_line(20 + i, "C1'", base, "S", i + 1, (7.0 * i, 18.0, 0.0)) for i, base in enumerate("GC")]
    path = tmp_path / "nt.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    meta = Polymer.from_pdb(str(path))
    assert int(meta.num_nucleotides) == 7
    _, by_chain = D.domain_restraint_pairs(meta, LOSS)
    part = lambda sl: SimpleNamespace(**{k: getattr(meta, k)[sl] for k in ("coord", "chain_id", "res_id")})
    aa = _direct_pairs(part(slice(0, 6)), meta.chain_id[:6])
    nt = {(i + 6, j + 6) for i, j in _direct_pairs(part(slice(6, None)), meta.chain_id[6:], 15.0, 15.0)}
    assert _as_set(by_chain) == aa | nt and any(meta.chain_id[i] != meta.chain_id[j] for i, j in nt)
    labels = np.unique(meta.chain_id, return_inverse=True)[1].reshape(-1)
    _, by_label = D.domain_restraint_pairs(meta, LOSS, labels)
    assert _as_set(by_label) == _direct_pairs(meta, labels) != _as_set(by_chain)
    plan = D.domain_plan(meta, LOSS, "a=A:1-6")
    assert plan["n_nt"] == 7


def test_plan_rows(meta):
    plan = D.domain_plan(meta, LOSS, "n=A:1-12; flex=A:13-20; c=A:21-30,B:12-20")
    rows = {r["name"]: r for r in plan["rows"]}
    assert list(rows) == ["n", "flex", "c", "rest of chain B"]
    assert rows["flex"]["beads"] == 6 and rows["flex"]["pairs"] == 0 and rows["flex"]["residues"] == "A:13-20"
    assert rows["rest of chain B"]["residues"] == "B:1-11" and rows["rest of chain B"]["beads"] == 11
    assert sum(r["pairs"] for r in plan["rows"]) == plan["domain_pairs"]["total"]
    assert plan["chain_pairs"]["total"] == len(_direct_pairs(meta, meta.chain_id))
    assert plan["connect_pairs"] == 45 and plan["n_nt"] == 0 and (plan["n_groups"], plan["n_single"]) == (3, 6)
    with pytest.raises(D.DomainSpecError, match="not both"):
        D.domain_plan(meta, LOSS, "n=A:1-12", "x.npy")


def test_plan_rows_from_file_singletons_and_cap(meta, tmp_path):
    n = len(meta.chain_id)
    labels = np.arange(n)
    labels[:10] = 1000  # one group of 10; everything else single beads
    np.save(tmp_path / "l.npy", labels)
    plan = D.domain_plan(meta, LOSS, file=tmp_path / "l.npy")
    assert [r["name"] for r in plan["rows"]] == ["label 1000", "single-bead groups"]
    assert plan["rows"][0]["residues"] == "A:1-10" and plan["rows"][1]["beads"] == n - 10
    assert plan["rows"][1]["residues"] == "A:11-15, A:18-30, B:1-20"
    pairs_rows = [dict(name=f"g{i}", kind="domain", labels=[i], spec="x") for i in range(5)]
    rows = D.domain_table_rows(pairs_rows, np.arange(5), np.array(list("AAAAA")), np.arange(1, 6),
                               {i: i for i in range(5)}, max_rows=3)
    assert [r["name"] for r in rows] == ["g0", "g1", "g2", "2 more groups"] and rows[-1]["pairs"] == 7


# ---------- train_atom.py's log ----------
def test_check_log():
    log = ("02/01 10:00:00 - mmengine - INFO - use domain instead of chain!\n"
           "02/01 10:00:00 - mmengine - INFO - found 1467 connect_pairs\n"
           "02/01 10:00:00 - mmengine - INFO - found 17663 cutoff_pairs\n"
           "02/01 10:00:00 - mmengine - INFO - found 3 clash_pairs\n")
    assert D.domain_check_log(log, 17663) == dict(used_domain=True, found=[17663], ok=True)
    assert D.domain_check_log(log, 19550)["ok"] is False
    no_domain = log.replace("use domain instead of chain!", "Reference structure has 1470 atom coordinates")
    assert D.domain_check_log(no_domain, 17663) == dict(used_domain=False, found=[17663], ok=False)
    assert D.domain_check_log("use domain instead of chain!\n", 5) == dict(used_domain=True, found=[], ok=False)
    assert D.domain_check_log("use domain instead of chain!\ncutoff_pairs is empty\n", 0)["ok"] is True


# ---------- mmengine keeps the base config's other keys ----------
def test_mmengine_merge_keeps_base_keys(tmp_path, model_pdb, meta):
    from mmengine import Config
    shutil.copy(EXAMPLE_CFG, tmp_path / "example.py")
    labels = D.domain_labels(D.domain_parse("n=A:1-12; flex=A:13-20"), meta.chain_id, meta.res_id)["labels"]
    domain_path = D.domain_save(labels, tmp_path / "prep")
    # the format the notebook's write_config writes
    (tmp_path / "child.py").write_text(
        '_base_ = "./example.py"\n\n'
        f"dataset_attr = dict(\n    ref_pdb_path={str(model_pdb)!r},\n)\n"
        f"extra_input_data_attr = dict(\n    use_domain=True,\n    domain_path={str(domain_path)!r},\n)\n")
    (tmp_path / "cont.py").write_text(
        '_base_ = "./example.py"\n\n'
        "extra_input_data_attr = dict(\n    ckpt_path='/runs/a/0006_0000100/ckpt.pt',\n    use_domain=True,\n"
        f"    domain_path={str(domain_path)!r},\n)\n")
    base = Config.fromfile(str(tmp_path / "example.py")).extra_input_data_attr
    child = Config.fromfile(str(tmp_path / "child.py"))
    attr = child.extra_input_data_attr
    assert set(attr) == set(base) and attr.use_domain is True and attr.domain_path == str(domain_path)
    assert attr.nma_path == base.nma_path == "" and attr.ckpt_path is None
    cont = Config.fromfile(str(tmp_path / "cont.py")).extra_input_data_attr
    assert cont.ckpt_path == "/runs/a/0006_0000100/ckpt.pt" and cont.use_domain is True and cont.nma_path == ""
    # the prediction 3.2 compares with the log, from the full config as train_atom.py reads it
    assert D.domain_predicted_pairs(child) == len(_direct_pairs(meta, labels))


# ---------- the labels belong to the model they were built for ----------
def test_model_key(meta):
    """3.2 refuses to train if the model's beads changed after 3.1 built the labels; moving the model is fine."""
    beads = lambda **k: SimpleNamespace(**{"chain_id": meta.chain_id, "res_id": meta.res_id,
                                           "atom_name": meta.atom_name, "coord": meta.coord, **k})
    key = D.domain_model_key(meta)
    assert key == D.domain_model_key(beads()) == D.domain_model_key(beads(coord=meta.coord + 25.0))
    order = np.r_[1, 0, 2:len(meta.chain_id)]
    variants = {
        "trimmed": SimpleNamespace(**{k: getattr(meta, k)[:-1] for k in ("chain_id", "res_id", "atom_name")}),
        "chain renamed": beads(chain_id=np.where(meta.chain_id == "B", "C", meta.chain_id)),
        "renumbered": beads(res_id=meta.res_id + (meta.chain_id == "B")),
        "reordered": SimpleNamespace(**{k: getattr(meta, k)[order] for k in ("chain_id", "res_id", "atom_name")}),
        "atom name": beads(atom_name=np.where(np.arange(len(meta.atom_name)) == 3, "C1'", meta.atom_name)),
    }
    for what, m in variants.items():
        assert D.domain_model_key(m) != key, what


def test_predicted_pairs_label_count(tmp_path, model_pdb, meta):
    """train_atom.py indexes the labels by bead without checking their number; the prediction refuses a mismatch."""
    n = len(meta.chain_id)
    for k in (n - 1, n + 1):
        path = D.domain_save(np.zeros(k, dtype=np.int64), tmp_path)
        cfg = dict(dataset_attr=dict(ref_pdb_path=str(model_pdb)), loss=LOSS,
                   extra_input_data_attr=dict(use_domain=True, domain_path=str(path)))
        with pytest.raises(D.DomainSpecError, match=f"{k:,} domain labels for {n:,} beads"):
            D.domain_predicted_pairs(cfg)
    cfg["extra_input_data_attr"]["domain_path"] = str(D.domain_save(np.zeros(n, dtype=np.int64), tmp_path))
    assert D.domain_predicted_pairs(cfg) == len(_direct_pairs(meta, np.zeros(n)))


# ---------- the notebook embeds this module verbatim ----------
def test_notebook_embeds_module_verbatim():
    nb = json.loads((NOTEBOOKS / "CryoSTAR_Colab.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title 1.1"))
    assert (NOTEBOOKS / "cryostar_domains.py").read_text() in cell


# ---------- acceptance on a real model ----------
@pytest.mark.skipif(not os.environ.get("CRYOSTAR_DOMAIN_TEST_PDB"), reason="CRYOSTAR_DOMAIN_TEST_PDB is not set")
def test_acceptance_user_model():
    from mmengine import Config
    from cryostar.utils.polymer import Polymer
    meta = Polymer.from_pdb(os.environ["CRYOSTAR_DOMAIN_TEST_PDB"])
    loss = Config.fromfile(str(EXAMPLE_CFG)).loss
    plan = D.domain_plan(meta, loss, USER_SPEC)
    print(f"\nbeads {len(meta.chain_id)}; continuity pairs {plan['connect_pairs']}; restraint pairs grouped by chain "
          f"{plan['chain_pairs']['total']} {plan['chain_pairs']['by_chain']}; with domains "
          f"{plan['domain_pairs']['total']} {plan['domain_pairs']['by_chain']}")
    for r in plan["rows"]:
        print(f"  {r['name']:<10} {r['residues']:<24} {r['beads']:>4} beads {r['pairs']:>6,} pairs  {r['note']}")
    assert len(meta.chain_id) == 1470
    assert plan["connect_pairs"] == 1467
    assert plan["chain_pairs"]["total"] == 19550
    assert plan["domain_pairs"]["total"] == 17663
    assert plan["domain_pairs"]["cross_chain"] == 0 and plan["warnings"] == []
    assert (plan["n_groups"], plan["n_single"]) == (6, 39)
