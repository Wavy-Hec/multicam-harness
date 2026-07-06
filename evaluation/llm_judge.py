"""LLM-as-judge summarization scoring. Replaces ROUGE as the headline summarization metric —
SummEval (Fabbri et al. 2021) documents ROUGE's ~0.15 Spearman correlation with human judgement.

Two judges, both G-Eval-style (Liu et al. 2023), prompt structure borrowed from lmms-eval:

  compare()        — single call, Reason-then-Score: one sentence of rationale, then
                     <score>0.0–1.0</score>. Cheap (1 call/pair); the default.
  compare_claims() — claim-level factual alignment: decompose the reference into atomic
                     claims, verify each YES/PARTIAL/NO against the generated text,
                     score = (YES + 0.5*PARTIAL) / claims. ~1+N calls/pair but fully
                     interpretable (you see exactly which facts were missed).

Bias mitigations: explicit verbosity penalty, reference always shown first, continuous
0.0–1.0 numeric scale, temperature 0 (deterministic given a fixed judge model).

CLI: python -m evaluation.llm_judge <summarization_results.json> [--judge gpt-5.2] [--claims]
Input format = the per-category results JSON written by runner.py (entries with
generated_answer + reference_answer). Writes per-entry scores back alongside the input file.
"""
import argparse
import json
import logging
import os
import re

import tqdm

logger = logging.getLogger(__name__)


class ConceptualSimilarityError(Exception):
    """Raised when the LLM response cannot be parsed into a valid similarity score."""


DEFAULT_JUDGE = "gpt-5.2"

_judge = None  # lazy: importing this module must never require an API key


def _get_judge(model=None):
    global _judge
    if _judge is None or (model and _judge.model != model):
        from models.clients import TextLLM
        from models.cost import resolve_openai_key
        _judge = TextLLM(model or DEFAULT_JUDGE, api_key=resolve_openai_key())
    return _judge


# --- Single-call Reason-then-Score judge -------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the conceptual "
    "and semantic similarity of generated summaries against a reference text. "
    "Your task is to compare the Generated Text with the Reference Text and "
    "determine how well the core concepts, facts, and conclusions are preserved. "
    "Here is how you accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on whether the same core facts, entities, numerical values, causal "
    "claims, and key conclusions are present. Ignore surface differences in "
    "vocabulary, grammar, word order, or sentence length.\n"
    "- Consider synonyms and paraphrases as valid matches "
    "(e.g. '20%' and 'one-fifth' are equivalent).\n"
    "- Penalise the Generated Text if it introduces hallucinations (information "
    "absent from the Reference), omits critical facts, or contradicts the Reference.\n"
    "- Do NOT reward verbosity. A longer Generated Text is not better unless the "
    "extra content is supported by the Reference.\n"
    "- Provide a single similarity score that reflects conceptual fidelity."
)

_USER_PROMPT = (
    "Please evaluate the following summarization pair:\n\n"
    "Reference Text (ground truth):\n{ref}\n\n"
    "Generated Text (model output):\n{gen}\n\n"
    "First, write ONE sentence of feedback explaining your score and citing "
    "specific agreements or gaps (e.g. hallucinations, omissions).\n"
    "Then output your score as: <score>NUMBER</score>\n\n"
    "The NUMBER must be a decimal between 0.0 and 1.0, where:\n"
    "  1.00 = perfect - all core concepts captured, no hallucinations or omissions\n"
    "  0.75 = high    - primary concepts present, only minor details missing\n"
    "  0.50 = moderate - some overlap but at least one major concept missing or inaccurate\n"
    "  0.25 = low     - largely unrelated or contradicts the reference\n"
    "  0.00 = none    - no semantic overlap\n"
    "Scores may be any value in [0.00, 1.00], not just the anchors above.\n"
    "Do not place any text after the closing </score> tag."
)


def compare(reference, generated, judge_model=None):
    """Single LLM call, Reason-then-Score. Returns a float in [0, 1]."""
    llm = _get_judge(judge_model)
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_PROMPT.format(ref=reference.strip(), gen=generated.strip())}"
    response = llm.prompt(full_prompt)
    logger.debug("LLM raw output:\n%s", response)

    match = re.search(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>", response)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    # Fallback: last standalone float in [0,1] — "last" because rationale precedes the score.
    logger.warning("Score tag not found; falling back to last float in response.")
    floats = re.findall(r"\b(0\.\d+|1\.0|0\.0|1|0)\b", response)
    if floats:
        return max(0.0, min(1.0, float(floats[-1])))

    raise ConceptualSimilarityError(f"Could not parse a valid score from LLM output:\n{response}")


# --- Claim-level factual-alignment judge --------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are a precise information extraction assistant. "
    "Your only job is to decompose a text into its distinct atomic factual claims."
)

_EXTRACT_USER = (
    "Extract every distinct factual claim from the following text. "
    "Output ONLY a numbered list, one claim per line, with no preamble or commentary. "
    "Each claim must be a single, self-contained, verifiable statement.\n\n"
    "Text:\n{ref}"
)

_VERIFY_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating whether a factual "
    "claim from a reference text is supported by a generated summary. "
    "Here is how you accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Consider synonyms, paraphrases, and numerical equivalents as valid matches "
    "(e.g. '20%' and 'one-fifth' are equivalent).\n"
    "- Answer YES if the generated text fully supports the claim.\n"
    "- Answer PARTIAL if the generated text implies or partially supports the claim "
    "but omits specifics.\n"
    "- Answer NO if the claim is absent from or contradicted by the generated text.\n"
    "- Do NOT reward the generated text for containing extra information beyond the claim."
)

_VERIFY_USER = (
    "Generated Text:\n{gen}\n\n"
    "Claim to verify: {claim}\n\n"
    "First write ONE sentence explaining your verdict, then output exactly one of: "
    "<verdict>YES</verdict>, <verdict>PARTIAL</verdict>, or <verdict>NO</verdict>. "
    "Do not place any text after the closing tag."
)


def _extract_claims(reference, judge_model=None):
    llm = _get_judge(judge_model)
    response = llm.prompt(f"{_EXTRACT_SYSTEM}\n\n{_EXTRACT_USER.format(ref=reference.strip())}")
    claims = []
    for line in response.strip().splitlines():
        clean = re.sub(r"^\s*[\d]+[.)]\s*|^\s*[-*]\s*", "", line).strip()
        if clean:
            claims.append(clean)
    if not claims:
        raise ConceptualSimilarityError(f"Claim extraction returned no claims. Raw response:\n{response}")
    return claims


def _verify_claim(claim, generated, judge_model=None):
    llm = _get_judge(judge_model)
    response = llm.prompt(f"{_VERIFY_SYSTEM}\n\n{_VERIFY_USER.format(gen=generated.strip(), claim=claim)}")
    match = re.search(r"<verdict>\s*(YES|PARTIAL|NO)\s*</verdict>", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for verdict in ("YES", "PARTIAL", "NO"):
        if re.search(rf"\b{verdict}\b", response, re.IGNORECASE):
            logger.warning("Used fallback verdict detection for claim: %s", claim[:60])
            return verdict
    logger.warning("Could not parse verdict for claim '%s'; defaulting to NO.", claim[:60])
    return "NO"


def compare_claims(reference, generated, judge_model=None):
    """Claim-level judge. Returns (score, verdicts) where verdicts = [(claim, YES/PARTIAL/NO)]."""
    claims = _extract_claims(reference, judge_model)
    verdicts = [(c, _verify_claim(c, generated, judge_model)) for c in claims]
    yes = sum(1 for _, v in verdicts if v == "YES")
    partial = sum(1 for _, v in verdicts if v == "PARTIAL")
    score = max(0.0, min(1.0, (yes + 0.5 * partial) / len(claims)))
    return score, verdicts


# --- CLI over a results JSON ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-judge a summarization results JSON")
    parser.add_argument("results_json", help="per-category results JSON from runner.py "
                                             "(entries with generated_answer + reference_answer)")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="judge model (OpenAI id)")
    parser.add_argument("--claims", action="store_true",
                        help="use the claim-level judge (interpretable, ~1+N calls per pair) "
                             "instead of the single-call Reason-then-Score judge")
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    scores = {}
    for key, entry in tqdm.tqdm(data.items(), total=len(data)):
        gen = entry.get("generated_answer", "")
        ref = entry.get("reference_answer", "")
        if not gen or not ref:
            continue
        try:
            if args.claims:
                score, verdicts = compare_claims(ref, gen, args.judge)
                scores[key] = {"judge_score": score,
                               "claim_verdicts": [{"claim": c, "verdict": v} for c, v in verdicts]}
            else:
                scores[key] = {"judge_score": compare(ref, gen, args.judge)}
        except ConceptualSimilarityError as ex:
            logger.warning("skipping %s: %s", key, ex)

    vals = [s["judge_score"] for s in scores.values()]
    mean = sum(vals) / len(vals) if vals else 0.0
    mode = "claims" if args.claims else "rts"
    out_path = os.path.splitext(args.results_json)[0] + f"_judge_{mode}.json"
    with open(out_path, "w") as f:
        json.dump({"judge_model": args.judge, "mode": mode, "mean_score": mean,
                   "n": len(vals), "per_question": scores}, f, indent=4)
    print(f"mean judge score: {mean:.4f} over {len(vals)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
