"""Decentralized harness: per-camera query-conditioned text summaries (pass 1, VLM) followed by
text-only aggregation over the concatenated summaries (pass 2, LLM). The aggregator never sees
pixels — everything it knows arrives through the per-camera text bottleneck."""
from evaluation.scoring import extract_letter, extract_number
from harnesses.uniform import uniform_sampling_strategy


def decentralized_frames(video_paths, num_frames, frame_budget="per_camera"):
    """Per-camera frames dict, same sampling as uniform. per_camera: each camera gets the full
    budget; total: split the budget across cameras for strict frame parity."""
    per_cam = num_frames
    if frame_budget == "total":
        per_cam = max(1, num_frames // max(1, len(video_paths)))
    return uniform_sampling_strategy(video_paths, per_cam)


def summarize_cameras(vllm_client, frames_data, question, camera_names, summary_max_tokens=512):
    """Pass 1: one query-conditioned summary per camera. Returns ({label: summary}, context)."""
    summaries = {}
    for cam_name, frames in frames_data.items():
        label = camera_names.get(cam_name, cam_name)
        summaries[label] = vllm_client.summarize_camera(frames, question, label, summary_max_tokens)
    context = "\n\n".join(f"Camera {lbl}:\n{txt}" for lbl, txt in summaries.items())
    return summaries, context


def decentralized_answer(aggregator, context, question, task, candidates=None, log_reasoning=False):
    """Pass 2: answer the question from the concatenated per-camera summaries (text only).

    Returns {"answer": <parsed answer for scoring>, "reasoning": <text or None>}. With
    log_reasoning the aggregator is asked to justify its choice (naming the camera + evidence)
    before the final answer, so the full chain can be greybox-debugged; scoring still uses the
    parsed answer. Without it, the strict letter/number parsing rules are used (clean comparison).
    """
    header = (
        "You are given text summaries from multiple cameras observing the same scene at the "
        "same time. Base your answer ONLY on these summaries; do not invent details.\n\n"
        f"{context}\n\n"
    )
    if task == "mc":
        opts = f"{question}\n" + "".join(f"{c}\n" for c in candidates)
        if log_reasoning:
            instr = ("First, in 1-2 sentences, explain which camera(s) and what evidence support "
                     "your choice. Then on a FINAL separate line write exactly 'ANSWER: <letter>'.")
            raw = aggregator.prompt(header + opts + "\n" + instr, max_tokens=256)
            return {"answer": extract_letter(raw, candidates), "reasoning": raw.strip()}
        parsing_rule = (
            "You must only return the letter of the answer choice, and nothing else. "
            "Do not include any other symbols, information, text, or justification in your answer. "
            "For example, if the correct answer is 'a) ...', you must only return 'a'."
        )
        raw = aggregator.prompt(header + opts + f"\n[PARSING RULE]: {parsing_rule}", max_tokens=16)
        return {"answer": extract_letter(raw, candidates), "reasoning": None}
    if task == "counting":
        if log_reasoning:
            instr = ("First, in 1-2 sentences, explain which camera(s) contribute to the count. "
                     "Then on a FINAL separate line write exactly 'ANSWER: <number>'.")
            raw = aggregator.prompt(header + f"{question}\n\n{instr}", max_tokens=256)
            return {"answer": extract_number(raw), "reasoning": raw.strip()}
        parsing_rule = "You must only return a single number as your answer, and nothing else."
        raw = aggregator.prompt(header + f"{question}\n\n[PARSING RULE]: {parsing_rule}", max_tokens=16)
        return {"answer": raw.strip(), "reasoning": None}
    # summarization: the generated text is itself the answer (and its own reasoning)
    raw = aggregator.prompt(header + question, max_tokens=512).strip()
    return {"answer": raw, "reasoning": None}
