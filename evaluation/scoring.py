# Ported from Wavy-Hec/CVBench Video-R1/src/eval_thinking.py @ f65d6e043014b6e9090c32dec4893ebc14fa4320
# Ported from Wavy-Hec/CVBench bench/metrics.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Answer scoring + benchmark metric aggregation.

``extract_think`` / ``extract_answer`` / ``parse_choice`` / ``gt_choice`` are
vendored from ``Video-R1/src/eval_thinking.py`` so this repo parses and scores
answers identically to the reference harness. The reference has since fixed two
scoring faults, and this file tracks it: a body that names no option letter is
no longer stored verbatim as a prediction, and a tagless direct answer is now
recovered instead of abstaining. Keeping the two in step is the point — the
equivalence claim is only meaningful against the reference's current behaviour.

The rest aggregates per-question Result rows into the benchmark metrics (M1-M4).

Pure-stdlib (no numpy) so it runs anywhere, including the login node for the
CPU scoring-validation gate.
"""
import functools
import re
from collections import defaultdict
from statistics import mean as _mean, pstdev as _pstdev


def extract_think(text):
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Thinking checkpoints (e.g. Qwen3-VL-*-Thinking) open <think> inside the
    # generation prompt, so the decoded output holds only "trace...</think>".
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    return ""


def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# Explicit "the answer is X" style conclusions, used ONLY when the model never
# emitted an <answer> tag (e.g. a truncated trace). We take the LAST such match
# (closest to the model's conclusion). Scanning the whole trace for any bare
# A/B/C/D is wrong: it grabs prose like "a vehicle" / "options A to D", which
# fabricated a lucky 'A' on truncated runs and made 0/20 structural.
@functools.lru_cache(maxsize=None)
def _conclude_mc_re(letters):
    return re.compile(
        r"(?i)(?:final\s+answer|best\s+answer|correct\s+answer|the\s+answer|answer)\s*"
        r"(?:is|:|=|would\s+be)?\s*\(?([" + letters + r"])\b"
    )


_CONCLUDE_MC = _conclude_mc_re("ABCD")
_CONCLUDE_YN = re.compile(
    r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)?\s*\(?(yes|no)\b"
)


# An explicit refusal, as opposed to a real option. Checked AFTER the verbatim
# option match below, because "Cannot be determined" is a genuine option on some
# temporal questions and must score as that option there.
_REFUSAL = re.compile(
    r"(?i)^\W*(n/?a\b|none\s+of\s+the\s+(above|options|provided)|none\b|"
    r"cannot\s+be\s+determined|can(?:no|')t\s+(?:be\s+)?determin|unknown\b|"
    r"unanswerable\b|not\s+determinable|insufficient\b|unclear\b)")


def _body_to_letter(body, letters, options=None):
    """Map an <answer> body to a single option letter, or "" to abstain.

    The rule this replaces was ``re.search("(?i)\\b([ABCD])\\b", body)`` — first
    match wins, case-insensitive — which reads the English article "a" as option
    A, so 'None of the above... does not show a lady' scored as a confident A.
    It also took the first of several listed letters and stored non-letter prose
    as a prediction with abstained=False.
    """
    body = body.strip()
    # 1. a bare letter, with or without decoration: "A", "(A)", "A.", "A:"
    m = re.fullmatch(r"\W*([" + letters + r"])\W*", body, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2. the body quotes an option verbatim -> that option's letter
    if options:
        norm = re.sub(r"\W+", " ", body).strip().lower()
        norm_lead = re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", body)
        norm_lead = re.sub(r"\W+", " ", norm_lead).strip().lower()
        for i, opt in enumerate(options[:len(letters)]):
            otext = re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", str(opt))
            otext = re.sub(r"\W+", " ", otext).strip().lower()
            if otext and any(n == otext or n.startswith(otext)
                             for n in (norm, norm_lead)):
                return letters[i].upper()
    # 3. an explicit refusal -> abstain
    if _REFUSAL.match(body):
        return ""
    # 4. a standalone option letter. Case-insensitive EXCEPT lowercase "a"/"i",
    #    which are English words. Ambiguous if several.
    found = {m.group(1).upper() for m in re.finditer(r"\b([" + letters + r"])\b",
                                                     body, re.IGNORECASE)
             if not (m.group(1) in ("a", "i"))}
    if len(found) == 1:
        return found.pop()
    return ""


def _leading_letter_option(body, letters, options):
    """Map ``"A. <the option's own text>"`` to ``"A"``, else "".

    This is how a model answers in direct-answer mode: the letter AND the option
    text, with no tags. Safe to trust at any length precisely because the text
    after the letter has to BE that option, so a reasoning trace that merely
    opens with "A." cannot match.
    """
    if not options:
        return ""
    m = re.match(r"\W*([" + letters + r"])\s*[.):]\s*(.+)", body,
                 re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    letter = m.group(1).upper()
    i = letters.upper().index(letter)
    if i >= len(options):
        return ""
    otext = re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", str(options[i]))
    otext = re.sub(r"\W+", " ", otext).strip().lower()
    rest = re.sub(r"\W+", " ", m.group(2)).strip().lower()
    if otext and (rest == otext or rest.startswith(otext)):
        return letter
    return ""


def parse_choice(text, is_yesno, letters="ABCD", options=None):
    """Final answer = <answer>..</answer> if present. If the tag is missing
    (e.g. a truncated trace), fall back to an explicit "the answer is X"
    conclusion (last match), then to a tagless direct answer; otherwise abstain
    (return "") rather than grabbing an incidental letter from reasoning prose.

    ``options`` is optional and only sharpens the mapping: pass the record's
    option list to resolve bodies that quote an option instead of naming its
    letter, and to tell a refusal apart from an option that happens to read
    like one.
    """
    ans = extract_answer(text)
    if is_yesno:
        if ans:
            m = re.search(r"(?i)\b(yes|no)\b", ans)
            return m.group(1).capitalize() if m else ans.strip()
        ms = list(_CONCLUDE_YN.finditer(text))
        return ms[-1].group(1).capitalize() if ms else ""
    if ans:
        return _body_to_letter(ans, letters, options)
    ms = list(_conclude_mc_re(letters).finditer(text))
    if ms:
        return ms[-1].group(1).upper()
    # direct-answer mode: no tags at all, either a bare "B" or "B. <option text>"
    stripped = text.strip()
    if not stripped:
        return ""
    lead = _leading_letter_option(stripped, letters, options)
    if lead:
        return lead
    # anything else only on a SHORT response — running _body_to_letter over a
    # long trace would pick up incidental letters, the failure this guard exists
    # to avoid
    if len(stripped) <= 40:
        return _body_to_letter(stripped, letters, options)
    return ""


def gt_choice(answer, is_yesno, letters="ABCD"):
    a = answer.strip()
    if is_yesno:
        return a.capitalize()
    m = re.search(r"(?i)([" + letters + r"])", a)
    return m.group(1).upper() if m else a.upper()


def _pct(xs, q):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def _acc(rows):
    n = len(rows)
    c = sum(1 for r in rows if r.get("correct"))
    return {"correct": c, "total": n, "acc": (c / n if n else None)}


def _by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return {str(k): _acc(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def summarize(rows):
    """rows: list of Result dicts (a single method x backend, or pooled)."""
    lat = [r.get("latency_s") for r in rows if r.get("latency_s") is not None]
    intok = [r.get("input_tokens") for r in rows if r.get("input_tokens") is not None]
    outtok = [r.get("output_tokens") for r in rows if r.get("output_tokens") is not None]
    calls = [r.get("num_model_calls") for r in rows if r.get("num_model_calls") is not None]
    n = len(rows)
    return {
        "n": n,
        "overall": _acc(rows),                       # M1
        "by_task_type": _by(rows, "task_type"),
        "by_orig_num_cameras": _by(rows, "orig_num_cameras"),
        "by_source": _by(rows, "source"),
        "by_cap_answer_safe": _by(rows, "cap_answer_safe"),
        "latency_s": {                               # M2
            "p50": _pct(lat, 0.50), "p95": _pct(lat, 0.95),
            "mean": (sum(lat) / len(lat) if lat else None), "n": len(lat),
        },
        "tokens": {                                  # M3
            "input_mean": (sum(intok) / len(intok) if intok else None),
            "output_mean": (sum(outtok) / len(outtok) if outtok else None),
            "calls_mean": (sum(calls) / len(calls) if calls else None),
        },
        "abstain_rate": (sum(1 for r in rows if r.get("abstained")) / n if n else None),  # M4
        "errors": sum(1 for r in rows if r.get("error")),
    }


def summarize_by_method_backend(rows):
    """Group rows by (method, backend) and summarize each -> the headline table."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("method"), r.get("backend"))].append(r)
    return {f"{m}/{b}": summarize(v) for (m, b), v in sorted(g.items())}


# --- 4-pass mean +/- std (Table 1 + plot error bars) -------------------------
# A "pass" = one sampled generation at a fixed seed. Accuracy is computed WITHIN
# each pass, then we report mean +/- std over the passes (std = decoding variance).

def _pass_accs(rows, filt=None):
    """Per-pass accuracies over rows (optionally filtered), as a list."""
    g = defaultdict(lambda: [0, 0])  # pass_idx -> [correct, total]
    for r in rows:
        if filt is not None and not filt(r):
            continue
        pi = r.get("pass_idx")
        g[pi][1] += 1
        if r.get("correct"):
            g[pi][0] += 1
    accs = []
    for _, (c, n) in sorted(g.items(), key=lambda kv: str(kv[0])):
        if n:
            accs.append(c / n)
    return accs


def _mstd(accs):
    if not accs:
        return {"mean": None, "std": None, "n_passes": 0, "per_pass": []}
    return {"mean": _mean(accs), "std": (_pstdev(accs) if len(accs) > 1 else 0.0),
            "n_passes": len(accs), "per_pass": accs}


def summarize_passes(rows):
    """Like ``summarize`` but adds mean+/-std-over-passes for overall, by task_type,
    by orig_num_cameras, and the task_type x cameras cross-tab (Plot 4)."""
    tts = sorted({r.get("task_type") for r in rows}, key=str)
    cams = sorted({r.get("orig_num_cameras") for r in rows},
                  key=lambda x: (x is None, x))
    base = summarize(rows)
    base["overall_passes"] = _mstd(_pass_accs(rows))
    base["by_task_type_passes"] = {
        str(tt): _mstd(_pass_accs(rows, lambda r, tt=tt: r.get("task_type") == tt))
        for tt in tts}
    base["by_orig_num_cameras_passes"] = {
        str(c): _mstd(_pass_accs(rows, lambda r, c=c: r.get("orig_num_cameras") == c))
        for c in cams}
    base["by_task_camera_passes"] = {
        str(tt): {str(c): _mstd(_pass_accs(
            rows, lambda r, tt=tt, c=c: r.get("task_type") == tt
            and r.get("orig_num_cameras") == c)) for c in cams}
        for tt in tts}
    return base


def summarize_by_method_backend_passes(rows):
    """Group by (method, backend) and summarize_passes each -> Table 1 + plot data."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("method"), r.get("backend"))].append(r)
    return {f"{m}/{b}": summarize_passes(v) for (m, b), v in sorted(g.items())}


def format_summary(rows):
    out = []
    for key, s in summarize_by_method_backend(rows).items():
        ov = s["overall"]
        lat = s["latency_s"]
        out.append(f"\n=== {key} ===")
        acc = f'{ov["acc"]*100:.1f}%' if ov["acc"] is not None else "n/a"
        out.append(f'  overall: {ov["correct"]}/{ov["total"]} = {acc}   '
                   f'abstain={s["abstain_rate"]*100:.0f}%  errors={s["errors"]}'
                   if ov["total"] else "  (no rows)")
        if lat["p50"] is not None:
            out.append(f'  latency_s: p50={lat["p50"]:.1f} p95={lat["p95"]:.1f} mean={lat["mean"]:.1f}')
        if s["tokens"]["input_mean"] is not None:
            out.append(f'  tokens: in~{s["tokens"]["input_mean"]:.0f} out~{s["tokens"]["output_mean"]:.0f} '
                       f'calls~{s["tokens"]["calls_mean"]:.1f}')
        out.append("  by task_type: " + ", ".join(
            f'{k} {v["correct"]}/{v["total"]}' for k, v in s["by_task_type"].items()))
    return "\n".join(out)
