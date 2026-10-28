"""Restraints per domain instead of per chain in CryoSTAR stage 1 (DOMAINS and DOMAIN_FILE in the notebook's 3.1).

train_atom.py builds its elastic network ("cutoff pairs": beads closer than loss.intra_chain_cutoff, minus the
sequence neighbours that the continuity term holds) within each chain. With extra_input_data_attr.use_domain = True
it builds them within the groups of a per-bead label array instead, loaded from extra_input_data_attr.domain_path
(a .npy). Continuity and the adaptive relaxation (loss.dist_keep_ratio) still use the chains; the clash term (beads
4-10 Å apart in the model) ignores chains and labels alike. The paper doesn't describe this option.

This module turns DOMAINS text (or a DOMAIN_FILE) into such labels and predicts the pairs train_atom.py will find,
with CryoSTAR's own functions called as train_atom.py calls them. The notebook embeds this file verbatim in cell 1.1,
so it stays self-contained on Colab; cryostar is imported inside the functions because 1.1 runs before CryoSTAR is
installed. Tests: notebooks/tests/test_cryostar_domains.py.
"""
import hashlib
import re
from pathlib import Path

import numpy as np

DOMAIN_FLEX = "flex"  # reserved DOMAINS name: each of its residues is a group of its own, so it gets no restraints
DOMAIN_LOG_LINE = "use domain instead of chain!"  # what train_atom.py logs when it uses the labels
_DOMAIN_FOUND_RE = re.compile(r"found (\d+) cutoff_pairs")
_DOMAIN_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*")
_DOMAIN_SEGMENT_RE = re.compile(r"([^\s:,;=]+)\s*:\s*(-?\d+)(?:\s*[-–]\s*(-?\d+))?")


class DomainSpecError(ValueError):
    """DOMAINS or DOMAIN_FILE can't be used; the message says why."""


def _domain_seg_text(chain, start, end):
    return f"{chain}:{start}" if start == end else f"{chain}:{start}-{end}"


def domain_parse(text):
    """Parse 'name=chain:start-end[,chain:start-end]; …' into [dict(name, flex, segments=[(chain, start, end)])].

    A segment can be a single residue ('C:300'). Whitespace is ignored, and empty entries and segments (a trailing ';'
    or ',') are skipped. The name 'flex' (any case) may appear more than once. Raises DomainSpecError on malformed text,
    start > end, a name used twice and segments that overlap (within an entry or between entries, flex included).
    """
    entries, names = [], {}
    for raw in str(text).split(";"):
        entry = raw.strip()
        if not entry:
            continue
        name, eq, body = (s.strip() for s in entry.partition("="))
        if not eq:
            raise DomainSpecError(f"'{entry}' has no '='. Write each domain as name=chain:start-end, "
                                  "e.g. L27=C:1-107, and separate domains with ';'.")
        if not _DOMAIN_NAME_RE.fullmatch(name):
            raise DomainSpecError(f"'{entry}': " + (f"'{name}' is not a usable name: use letters, digits, '_', '.', "
                                                    "'+' or '-'." if name else "the domain has no name."))
        flex = name.lower() == DOMAIN_FLEX
        if not flex:
            if name.lower() in names:
                first = names[name.lower()]
                raise DomainSpecError(f"The name {name} is used twice" + (f" ({first} and {name})" if first != name
                                                                          else "")
                                      + ". Give each domain one entry and separate its segments with ',' "
                                      "(only flex can repeat).")
            names[name.lower()] = name
        segments = []
        for part in body.split(","):
            seg = part.strip()
            if not seg:
                continue
            m = _DOMAIN_SEGMENT_RE.fullmatch(seg)
            if not m:
                hint = (f" Give the chain too, e.g. C:{seg}." if re.fullmatch(r"-?\d+(\s*[-–]\s*-?\d+)?", seg)
                        else " Write it as chain:start-end (e.g. C:1-107) or chain:residue (e.g. C:300).")
                raise DomainSpecError(f"'{seg}' in '{entry}' is not a residue range.{hint}")
            chain, start = m.group(1), int(m.group(2))
            end = start if m.group(3) is None else int(m.group(3))
            if start > end:
                raise DomainSpecError(f"'{seg}' in {name}: the range starts after it ends.")
            segments.append((chain, start, end))
        if not segments:
            raise DomainSpecError(f"'{entry}' lists no residues.")
        entries.append(dict(name=DOMAIN_FLEX if flex else name, flex=flex, segments=segments))
    if str(text).strip() and not entries:
        raise DomainSpecError("DOMAINS lists no domains.")
    # Sorted by start, any overlap shows up between neighbours.
    flat = sorted((c, s, e, i) for i, d in enumerate(entries) for c, s, e in d["segments"])
    for (c1, s1, e1, i1), (c2, s2, e2, i2) in zip(flat, flat[1:]):
        if c1 == c2 and s2 <= e1:
            a, b = entries[i1]["name"], entries[i2]["name"]
            both = f"{a} ({_domain_seg_text(c1, s1, e1)}) and {b} ({_domain_seg_text(c2, s2, e2)})" if i1 != i2 \
                else f"{_domain_seg_text(c1, s1, e1)} and {_domain_seg_text(c2, s2, e2)} in {a}"
            raise DomainSpecError(f"{both} overlap at {_domain_seg_text(c1, s2, min(e1, e2))}. "
                                  "Each residue can belong to one domain only.")
    return entries


def _domain_runs(values):
    """Sorted unique integers as [(start, end)] runs of consecutive numbers."""
    values = np.unique(np.asarray(values, dtype=np.int64))
    if not len(values):
        return []
    breaks = np.flatnonzero(np.diff(values) != 1)
    starts = np.concatenate(([values[0]], values[breaks + 1]))
    ends = np.concatenate((values[breaks], [values[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def domain_segments(chain_id, res_id, mask, limit=6):
    """The residues selected by `mask`, as text such as 'C:1-107, C:325-540' (chains in model order)."""
    chain_id, res_id, mask = np.asarray(chain_id), np.asarray(res_id), np.asarray(mask, dtype=bool)
    parts = []
    for c in dict.fromkeys(chain_id[mask].tolist()):
        parts += [_domain_seg_text(c, s, e) for s, e in _domain_runs(res_id[mask & (chain_id == c)])]
    more = len(parts) - limit
    return ", ".join(parts[:limit]) + (f" and {more} more range(s)" if more > 0 else "")


def domain_labels(entries, chain_id, res_id):
    """Per-bead labels for parsed DOMAINS, in the model's bead order (Polymer.from_pdb).

    Each domain is one group; each residue of a flex entry is a group of its own; the residues of a chain that no
    entry lists form one group per chain. Labels count from 0 in order of first appearance along the beads.
    Returns dict(labels=int64 array, groups=[dict(name, kind, labels, spec)], warnings=[str]); kind is 'domain',
    'flex' (all flex entries together) or 'chain'. Raises DomainSpecError for an unknown chain and for a domain
    none of whose residues is in the model; residues listed but absent from the model give a warning.
    """
    chain_id, res_id = np.asarray(chain_id), np.asarray(res_id)
    chains = list(dict.fromkeys(chain_id.tolist()))
    member = np.full(len(chain_id), -1)
    warnings = []
    for i, d in enumerate(entries):
        hit = 0
        for c, s, e in d["segments"]:
            if c not in chains:
                raise DomainSpecError(f"{d['name']}: the model has no chain {c} (its chains are {', '.join(chains)}).")
            sel = (chain_id == c) & (res_id >= s) & (res_id <= e)
            member[sel] = i
            hit += int(sel.sum())
            present = np.unique(res_id[sel])
            missing = (e - s + 1) - len(present)
            if missing:
                edges = np.concatenate(([s - 1], present, [e + 1]))
                gaps = [_domain_seg_text(c, int(a) + 1, int(b) - 1) for a, b in zip(edges, edges[1:]) if b - a > 1]
                more = f" and {len(gaps) - 3} more range(s)" if len(gaps) > 3 else ""
                warnings.append(f"{d['name']}: {missing:,} residue(s) of {_domain_seg_text(c, s, e)} are not in the "
                                f"model ({', '.join(gaps[:3])}{more}).")
        if hit == 0:
            raise DomainSpecError(f"None of the residues of {d['name']} "
                                  f"({', '.join(_domain_seg_text(*g) for g in d['segments'])}) is in the model.")
    keys, bead_chain = {}, chain_id.tolist()
    labels = np.empty(len(chain_id), dtype=np.int64)
    for k, i in enumerate(member.tolist()):
        key = ("chain", bead_chain[k]) if i < 0 else ("flex", k) if entries[i]["flex"] else ("domain", i)
        labels[k] = keys.setdefault(key, len(keys))
    groups, flex = [], None
    for i, d in enumerate(entries):
        spec = ", ".join(_domain_seg_text(*g) for g in d["segments"])
        if d["flex"]:
            if flex is None:
                flex = dict(name=DOMAIN_FLEX, kind="flex", labels=[], spec=spec)
                groups.append(flex)
            else:
                flex["spec"] += ", " + spec
        else:
            groups.append(dict(name=d["name"], kind="domain", labels=[keys[("domain", i)]], spec=spec))
    if flex is not None:
        flex["labels"] = sorted(v for k, v in keys.items() if k[0] == "flex")
    listed = {c for d in entries for c, _, _ in d["segments"]}
    for c in chains:
        if ("chain", c) in keys:
            groups.append(dict(name=f"rest of chain {c}" if c in listed else f"chain {c}", kind="chain",
                               labels=[keys[("chain", c)]], spec=None))
    return dict(labels=labels, groups=groups, warnings=warnings)


def domain_labels_from_file(path, n_beads):
    """Labels from a ready-made .npy: 1-D, integers, one per bead. Returns dict(labels, groups, warnings) like
    domain_labels, with one group per label value."""
    p = Path(str(path).strip()).expanduser()
    if not p.is_file():
        raise DomainSpecError(f"DOMAIN_FILE {p} does not exist.")
    try:
        a = np.load(p, allow_pickle=False)
    except Exception as e:
        raise DomainSpecError(f"{p.name} can't be read as a .npy file without pickle ({type(e).__name__}: {e}).")
    if hasattr(a, "files"):
        a.close()
        raise DomainSpecError(f"{p.name} is an .npz archive; give a .npy file holding one array of labels.")
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise DomainSpecError(f"{p.name} holds a {a.dtype} array of shape {a.shape}; it needs one integer label per "
                              "bead (a 1-D array; save it with np.save(path, labels.astype(np.int64))).")
    if len(a) != n_beads:
        raise DomainSpecError(f"{p.name} has {len(a):,} labels, but the model has {n_beads:,} beads (one per residue: "
                              "Cα for protein, C1′ for nucleic acid, in the model's order).")
    labels = a.astype(np.int64)
    values = list(dict.fromkeys(labels.tolist()))
    groups = [dict(name=f"label {v}", kind="file", labels=[v], spec=None) for v in values]
    return dict(labels=labels, groups=groups, warnings=[])


def domain_restraint_pairs(meta, loss, domain_id=None):
    """(connect_pairs, cutoff_pairs) as CryoEMTask.__init__ in train_atom.py builds them, step by step.

    meta: a Polymer (or anything with coord, chain_id, res_id and atom_name arrays); loss: the config's `loss`;
    domain_id: the labels when extra_input_data_attr.use_domain is on, else None. With labels, train_atom.py applies
    the protein cutoffs to every bead (its separate nucleotide branch is skipped). If no cutoff pair is found, the
    empty array is returned as is: train_atom.py itself then fails in remove_duplicate_pairs.
    """
    from cryostar.utils.dist_loss import find_continuous_pairs, find_quaint_cutoff_pairs, remove_duplicate_pairs
    from cryostar.utils.polymer import AA_ATOMS, NT_ATOMS
    connect_pairs = find_continuous_pairs(meta.chain_id, meta.res_id, meta.atom_name)
    if domain_id is not None:
        cutoff_pairs = find_quaint_cutoff_pairs(meta.coord, domain_id, meta.res_id, loss["intra_chain_cutoff"],
                                                loss["inter_chain_cutoff"], loss["intra_chain_res_bound"])
    elif np.sum(np.isin(meta.atom_name, NT_ATOMS)):
        tmp_mask = np.isin(meta.atom_name, AA_ATOMS)
        indices_in_pdb = np.nonzero(tmp_mask)[0]
        aa_cutoff_pairs = find_quaint_cutoff_pairs(meta.coord[tmp_mask], meta.chain_id[tmp_mask],
                                                   meta.res_id[tmp_mask], loss["intra_chain_cutoff"],
                                                   loss["inter_chain_cutoff"], loss["intra_chain_res_bound"])
        aa_cutoff_pairs = indices_in_pdb[aa_cutoff_pairs]
        tmp_mask = np.isin(meta.atom_name, NT_ATOMS)
        indices_in_pdb = np.nonzero(tmp_mask)[0]
        nt_cutoff_pairs = find_quaint_cutoff_pairs(meta.coord[tmp_mask], meta.chain_id[tmp_mask],
                                                   meta.res_id[tmp_mask], loss["nt_intra_chain_cutoff"],
                                                   loss["nt_inter_chain_cutoff"], loss["nt_intra_chain_res_bound"])
        nt_cutoff_pairs = indices_in_pdb[nt_cutoff_pairs]
        cutoff_pairs = np.vstack((aa_cutoff_pairs, nt_cutoff_pairs))
    else:
        cutoff_pairs = find_quaint_cutoff_pairs(meta.coord, meta.chain_id, meta.res_id, loss["intra_chain_cutoff"],
                                                loss["inter_chain_cutoff"], loss["intra_chain_res_bound"])
    if len(cutoff_pairs) == 0:
        return connect_pairs, cutoff_pairs
    cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, connect_pairs)
    if loss["sse_weight"] != 0.0:
        sse_pairs = find_quaint_cutoff_pairs(meta.coord, meta.chain_id, meta.res_id, loss["intra_chain_cutoff"], 0, 20)
        cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, sse_pairs)
    return connect_pairs, cutoff_pairs


def domain_pair_counts(pairs, chain_id, labels=None):
    """Pairs per chain (both beads in it), between chains, and (with labels) per label and between labels."""
    pairs, chain_id = np.asarray(pairs).reshape(-1, 2), np.asarray(chain_id)
    ci, cj = chain_id[pairs[:, 0]], chain_id[pairs[:, 1]]
    same = ci == cj
    out = dict(total=len(pairs), cross_chain=int(np.sum(~same)),
               by_chain={c: int(np.sum(same & (ci == c))) for c in dict.fromkeys(chain_id.tolist())})
    if labels is not None:
        labels = np.asarray(labels)
        li, lj = labels[pairs[:, 0]], labels[pairs[:, 1]]
        values, counts = np.unique(li[li == lj], return_counts=True)
        out["by_label"] = dict(zip(values.tolist(), counts.tolist()))
        out["between_labels"] = int(np.sum(li != lj))
    return out


def domain_table_rows(groups, labels, chain_id, res_id, by_label, max_rows=40):
    """Rows (name, residues, beads, pairs, note) for the domain table. Single-bead groups from a file are summed in
    one row, like flex; past max_rows the remaining groups are summed in a last row."""
    labels = np.asarray(labels)
    sizes = dict(zip(*(a.tolist() for a in np.unique(labels, return_counts=True))))
    rows, singles = [], []
    for g in groups:
        if g["kind"] == "file" and sizes[g["labels"][0]] == 1:
            singles += g["labels"]
            continue
        mask = np.isin(labels, g["labels"])
        rows.append(dict(name=g["name"], residues=g["spec"] or domain_segments(chain_id, res_id, mask),
                         beads=int(mask.sum()), pairs=sum(by_label.get(v, 0) for v in g["labels"]),
                         note="each residue on its own" if g["kind"] == "flex" else ""))
    if len(rows) > max_rows:
        rest = rows[max_rows:]
        rows = rows[:max_rows] + [dict(name=f"{len(rest):,} more groups", residues="", beads=sum(r["beads"] for r in rest),
                                       pairs=sum(r["pairs"] for r in rest), note="")]
    if singles:
        mask = np.isin(labels, singles)
        rows.append(dict(name="single-bead groups", residues=domain_segments(chain_id, res_id, mask),
                         beads=int(mask.sum()), pairs=sum(by_label.get(v, 0) for v in singles),
                         note="each residue on its own"))
    return rows


def domain_plan(meta, loss, text="", file=""):
    """Everything 3.1 shows and writes for DOMAINS text or a DOMAIN_FILE (not both), for the model `meta`.

    Returns dict(labels, groups, warnings, rows, chain_pairs, domain_pairs (counts as from domain_pair_counts),
    connect_pairs, n_groups (of 2 or more beads), n_single (single-bead groups), n_nt, beads_by_chain). Raises DomainSpecError if the input can't be used, or if
    train_atom.py would fail with these labels (no restraint pair at all, or none within a chain).
    """
    text, file = str(text or "").strip(), str(file or "").strip()
    if text and file:
        raise DomainSpecError("Set DOMAINS or DOMAIN_FILE, not both.")
    if file:
        plan = domain_labels_from_file(file, len(meta.chain_id))
    else:
        plan = domain_labels(domain_parse(text), meta.chain_id, meta.res_id)
    from cryostar.utils.polymer import NT_ATOMS
    sizes = np.unique(plan["labels"], return_counts=True)[1]
    connect, chain_pairs = domain_restraint_pairs(meta, loss)
    _, domain_pairs = domain_restraint_pairs(meta, loss, plan["labels"])
    plan.update(chain_pairs=domain_pair_counts(chain_pairs, meta.chain_id),
                domain_pairs=domain_pair_counts(domain_pairs, meta.chain_id, plan["labels"]),
                connect_pairs=len(connect), n_groups=int(np.sum(sizes > 1)), n_single=int(np.sum(sizes == 1)),
                n_nt=int(np.sum(np.isin(meta.atom_name, NT_ATOMS))),
                beads_by_chain={c: int(np.sum(meta.chain_id == c)) for c in dict.fromkeys(meta.chain_id.tolist())})
    counts = plan["domain_pairs"]
    if counts["total"] == 0:
        raise DomainSpecError("With these domains no restraint pair is left (every group is too small), and "
                              "CryoSTAR's train_atom.py fails without any. Make at least one domain larger.")
    if counts["total"] == counts["cross_chain"]:
        raise DomainSpecError("Every restraint pair with these domains joins two different chains; CryoSTAR's "
                              "train_atom.py needs at least one within a chain (for dist_keep_ratio) and fails "
                              "otherwise.")
    plan["rows"] = domain_table_rows(plan["groups"], plan["labels"], meta.chain_id, meta.res_id, counts["by_label"])
    return plan


def domain_predicted_pairs(cfg):
    """The number of cutoff pairs train_atom.py finds for a full config (e.g. a run's config.py). Raises
    DomainSpecError if the labels don't have one entry per bead (train_atom.py doesn't check this)."""
    from cryostar.utils.polymer import Polymer
    meta = Polymer.from_pdb(str(cfg["dataset_attr"]["ref_pdb_path"]))
    attr = cfg.get("extra_input_data_attr") or {}
    labels = np.load(str(attr["domain_path"])) if attr.get("use_domain") else None
    if labels is not None and len(labels) != len(meta.chain_id):
        raise DomainSpecError(f"{len(labels):,} domain labels for {len(meta.chain_id):,} beads")
    return len(domain_restraint_pairs(meta, cfg["loss"], labels)[1])


def domain_model_key(meta):
    """Fingerprint of a model's beads (their order, chains, residue numbers and atom names), which per-bead labels
    depend on. Coordinates are left out: moving the model doesn't misalign the labels."""
    beads = zip(np.asarray(meta.chain_id).tolist(), np.asarray(meta.res_id).tolist(),
                np.asarray(meta.atom_name).tolist())
    return hashlib.sha256(repr(list(beads)).encode()).hexdigest()


def domain_check_log(text, predicted):
    """Did train_atom.py use the labels (its log line) and find `predicted` cutoff pairs? Returns
    dict(used_domain, found=[numbers it logged], ok)."""
    found = [int(n) for n in _DOMAIN_FOUND_RE.findall(text)]
    if "cutoff_pairs is empty" in text:
        found.append(0)
    used = DOMAIN_LOG_LINE in text
    return dict(used_domain=used, found=found, ok=used and bool(found) and all(n == predicted for n in found))


def domain_same_grouping(a, b):
    """True if two label arrays group the beads the same way (whatever numbers they use)."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        return False

    def canonical(x):
        _, first, inverse = np.unique(x, return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=np.int64)
        rank[np.argsort(first)] = np.arange(len(first))
        return rank[inverse.reshape(-1)]
    return bool(np.array_equal(canonical(a), canonical(b)))


def domain_save(labels, folder):
    """Save labels as int64 .npy named after their content (domains_<hash>.npy); unchanged files are not rewritten."""
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    path = Path(folder) / f"domains_{hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest()[:12]}.npy"
    try:
        same = path.is_file() and np.array_equal(np.load(path, allow_pickle=False), labels)
    except Exception:
        same = False
    if not same:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, labels)
    return path
