"""Deterministic scoring: answer parsing, exact-match MC, counting partial credit, ROUGE."""
import re

from rouge_score import rouge_scorer


def extract_letter(text, candidates):
    """Pull the chosen option letter out of a (possibly verbose) aggregator response."""
    out = text.lower()
    m = re.search(r"answer\s*[:=]\s*\(?([a-z])\)?", out)  # prefer an explicit 'ANSWER: x' line
    if m:
        return m.group(1)
    valid = {str(c).strip()[0].lower() for c in (candidates or []) if str(c).strip()}
    letters = re.findall(r"\b([a-z])\b", out)
    for l in letters:
        if l in valid:
            return l
    return letters[0] if letters else out.strip()[:1]


def extract_number(text):
    m = re.search(r"answer\s*[:=]\s*(-?\d+\.?\d*)", text.lower())
    if m:
        return m.group(1)
    m = re.search(r"-?\d+\.?\d*", text)
    return m.group() if m else text.strip()


def counting_partial_credit(predicted, correct_answer):
    def num(s):
        m = re.search(r"-?\d+\.?\d*", str(s))
        return float(m.group()) if m else None
    pred, gt = num(predicted), num(correct_answer)
    if gt is None or pred is None:
        return 0.0
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1 - abs(pred - gt) / abs(gt))


def rouge(reference, generated):
    """ROUGE-1/2/L f-measures. Kept for continuity/lexical sanity checks; the LLM judge
    (evaluation.llm_judge) is the preferred summarization metric — ROUGE is blind to
    paraphrase and rewards surface overlap."""
    raw = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(reference, generated)
    return {k: raw[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}
