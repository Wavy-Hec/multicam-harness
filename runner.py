"""Experiment loop: iterate questions, dispatch to the chosen harness, score, write results.

Kept harness-agnostic on purpose — the only place strategy matters is (a) how frames are
packaged (harnesses.get_frames) and (b) whether answering goes through the decentralized
two-pass path. Question text, scoring, and output format are identical across harnesses.
"""
import json
import os
import time

import tqdm

from dataloaders.qa_json import COUNTING_CATS, SUMMARIZATION_CATS
from evaluation.scoring import counting_partial_credit, rouge
from harnesses import get_frames
from harnesses.decentralized import decentralized_answer, summarize_cameras
from models.cost import METER, BudgetExceeded

# Output base dir. Default = repo root (results/). --out_dir overrides it so parallel
# shards / separate runs never overwrite each other (a fixed per-model path otherwise clobbers).
_OUT_BASE = None


def set_out_base(path):
    global _OUT_BASE
    _OUT_BASE = os.path.abspath(path)


def _out_base():
    return _OUT_BASE if _OUT_BASE else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_category_experiment(category_name, category_dataset, vllm_client, model_name, num_frames, dataset, strategy,
                            frame_budget="per_camera", aggregator=None, summary_max_tokens=512, pass_tag="",
                            log_reasoning=False):
    results, correct, total, rouge_scores, counting_scores = {}, 0, 0, [], []
    halted = False
    print(f"\nProcessing '{category_name}' with {strategy} strategy...")
    for key in tqdm.tqdm(list(category_dataset.keys()), desc=category_name):
        entry = category_dataset[key]
        if not entry.get("video_paths"):
            continue
        camera_names = entry.get("camera_names", {})
        frames_data = get_frames(strategy, entry, num_frames, frame_budget)
        if not frames_data:
            continue

        question, candidates, correct_answer = entry["question"], entry["candidates"], entry["correct_answer"]
        qtype = entry.get("question_type", category_name)
        t0 = time.time()
        tok0p, tok0c, usd0 = METER.prompt_tokens, METER.completion_tokens, METER.usd
        try:
            # Decentralized pass 1: one query-conditioned summary per camera, then the
            # text-only aggregator (pass 2) answers from the concatenated summaries. For
            # uniform/stitched, summaries stays None and the VLM answers directly from frames.
            summaries, context, answer_reasoning = None, None, None
            if strategy == "decentralized":
                summaries, context = summarize_cameras(vllm_client, frames_data, question,
                                                       camera_names, summary_max_tokens)

            if qtype in COUNTING_CATS:
                if strategy == "decentralized":
                    _res = decentralized_answer(aggregator, context, question, "counting", log_reasoning=log_reasoning)
                    predicted, answer_reasoning = _res["answer"], _res["reasoning"]
                else:
                    predicted = vllm_client.counting(frames_data, question, strategy, camera_names)
                pc = counting_partial_credit(predicted, correct_answer)
                exact = 1 if predicted.strip() == correct_answer.strip() else 0
                counting_scores.append(pc); total += 1
                results[key] = {"question": question, "predicted_answer": predicted,
                                "correct_answer": correct_answer, "partial_credit": pc,
                                "is_correct": exact, "question_type": qtype, "strategy": strategy}
            elif qtype in SUMMARIZATION_CATS:
                if strategy == "decentralized":
                    generated = decentralized_answer(aggregator, context, question, "summarize")["answer"]
                else:
                    generated = vllm_client.summarize(frames_data, question, strategy, camera_names)
                scores = rouge(correct_answer, generated)
                rouge_scores.append(scores); total += 1
                results[key] = {"question": question, "generated_answer": generated,
                                "reference_answer": correct_answer, "rouge_scores": scores,
                                "question_type": qtype, "strategy": strategy}
            else:
                if strategy == "decentralized":
                    _res = decentralized_answer(aggregator, context, question, "mc", candidates, log_reasoning=log_reasoning)
                    predicted, answer_reasoning = _res["answer"], _res["reasoning"]
                else:
                    predicted = vllm_client.multiple_choice(frames_data, question, candidates, strategy, camera_names)
                is_correct = 1 if predicted == correct_answer else 0
                correct += is_correct; total += 1
                results[key] = {"question": question, "predicted_answer": predicted,
                                "correct_answer": correct_answer, "is_correct": is_correct,
                                "question_type": qtype, "strategy": strategy}

            if summaries is not None:
                results[key]["camera_summaries"] = summaries
                if answer_reasoning is not None:
                    results[key]["answer_reasoning"] = answer_reasoning
        except BudgetExceeded:
            raise
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

        # per-question latency, token, cost, and greybox (needed-camera / GT-reasoning) accounting
        if key in results:
            results[key]["latency_s"] = round(time.time() - t0, 3)
            results[key]["tokens"] = {"prompt": METER.prompt_tokens - tok0p,
                                      "completion": METER.completion_tokens - tok0c}
            dusd = METER.usd - usd0
            if dusd > 0:
                results[key]["cost_usd"] = round(dusd, 6)
            # ground-truth camera provenance: which cameras were needed vs how many we summarized
            results[key]["needed_cameras"] = entry.get("needed_cameras")
            results[key]["num_needed_cameras"] = entry.get("num_needed_cameras")
            results[key]["num_cameras_summarized"] = len(summaries) if summaries is not None else None
            if log_reasoning and entry.get("gt_reasoning"):
                results[key]["gt_reasoning"] = entry.get("gt_reasoning")

        # hard cap: stop the run before exceeding --max_usd (partial results are still written)
        if METER.over_cap():
            print(f"\n[BUDGET] Hit ${METER.usd:.2f} (cap ${METER.max_usd:.2f}) — halting after '{key}'.")
            halted = True
            break

    model_name_clean = model_name.replace("/", "_")
    output_dir = os.path.join(_out_base(), model_name_clean, dataset, strategy)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset}_{strategy}_{category_name}{pass_tag}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    if rouge_scores:
        avg = {k: sum(s[k] for s in rouge_scores) / len(rouge_scores) for k in ("rouge1", "rouge2", "rougeL")}
        print(f"{category_name} avg ROUGE ({len(rouge_scores)}): R-1={avg['rouge1']:.4f} R-2={avg['rouge2']:.4f} R-L={avg['rougeL']:.4f}")
    elif counting_scores:
        pc = sum(counting_scores) / len(counting_scores)
        print(f"{category_name} partial-credit: {pc:.2%} ({len(counting_scores)})")
    else:
        acc = correct / total if total else 0
        print(f"{category_name} accuracy: {acc:.2%} ({correct}/{total})")
    if halted:
        raise BudgetExceeded(f"cost cap ${METER.max_usd:.2f} reached during '{category_name}'")
    return {"correct": correct, "total": total, "rouge_scores": rouge_scores,
            "counting_scores": counting_scores}


def _category_scalar(stats):
    """Reduce a category's stats to one scalar in [0,1] for cross-pass aggregation."""
    if stats.get("rouge_scores"):
        rs = stats["rouge_scores"]; return sum(s["rougeL"] for s in rs) / len(rs)
    if stats.get("counting_scores"):
        cs = stats["counting_scores"]; return sum(cs) / len(cs)
    return stats["correct"] / stats["total"] if stats.get("total") else 0.0


def _run_single_pass(datasets_by_category, vllm_client, model_name, num_frames, dataset, strategy,
                     frame_budget, aggregator, summary_max_tokens, pass_tag="", log_reasoning=False):
    all_stats, halted = {}, False
    for category_name, category_dataset in datasets_by_category.items():
        try:
            all_stats[category_name] = run_category_experiment(
                category_name, category_dataset, vllm_client, model_name, num_frames, dataset, strategy,
                frame_budget=frame_budget, aggregator=aggregator, summary_max_tokens=summary_max_tokens,
                pass_tag=pass_tag, log_reasoning=log_reasoning)
        except BudgetExceeded as e:
            print(f"[BUDGET] {e} — stopping remaining categories.")
            halted = True
            break
    return all_stats, halted


def _format_table(all_stats, strategy, num_frames, header_extra=""):
    lines = ["=" * 50, f"Final Results ({strategy} strategy, {num_frames} frames){header_extra}:", "=" * 50]
    mc_correct = mc_total = 0
    for category, stats in all_stats.items():
        if stats["rouge_scores"]:
            rs = stats["rouge_scores"]
            avg = {k: sum(s[k] for s in rs) / len(rs) for k in ("rouge1", "rouge2", "rougeL")}
            lines.append(f"{category:20s}: R-1={avg['rouge1']:.4f} R-2={avg['rouge2']:.4f} R-L={avg['rougeL']:.4f} ({len(rs)})")
        elif stats.get("counting_scores"):
            cs = stats["counting_scores"]
            lines.append(f"{category:20s}: {sum(cs) / len(cs):.2%} partial-credit ({len(cs)})")
        else:
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            lines.append(f"{category:20s}: {acc:.2%} ({stats['correct']}/{stats['total']})")
            mc_correct += stats["correct"]; mc_total += stats["total"]
    lines.append("=" * 50)
    if mc_total:
        lines.append(f"{'MC Overall':20s}: {mc_correct / mc_total:.2%} ({mc_correct}/{mc_total})")
    lines.append("=" * 50)
    return "\n".join(lines)


def _write_results_txt(model_name, dataset, table):
    results_dir = os.path.join(_out_base(), model_name.replace("/", "_"), dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.txt")
    mode = "a" if os.path.exists(results_file) else "w"
    with open(results_file, mode) as f:
        if mode == "a":
            f.write("\n\n")
        f.write(table + "\n")


def run_experiment(datasets_by_category, vllm_client, model_name, num_frames, dataset, strategy,
                   frame_budget="per_camera", aggregator=None, summary_max_tokens=512,
                   passes=1, seeds=(1,), temperature=None, log_reasoning=False):
    # Single pass (default): identical to before, with an optional temperature/seed override.
    if not passes or passes <= 1:
        if temperature is not None:
            vllm_client.temp_override = temperature; vllm_client.seed = seeds[0]
            if aggregator is not None:
                aggregator.temp_override = temperature; aggregator.seed = seeds[0]
        all_stats, _ = _run_single_pass(datasets_by_category, vllm_client, model_name, num_frames,
                                        dataset, strategy, frame_budget, aggregator, summary_max_tokens,
                                        log_reasoning=log_reasoning)
        table = _format_table(all_stats, strategy, num_frames)
        print("\n" + table); _write_results_txt(model_name, dataset, table)
        return

    # Multi-pass: run the whole eval once per seed (sampling temperature), aggregate mean +/- std.
    import statistics
    per_pass_scalars, per_pass_mc = [], []
    for pi in range(passes):
        seed = seeds[pi] if pi < len(seeds) else seeds[-1] + (pi - len(seeds) + 1)
        vllm_client.seed = seed; vllm_client.temp_override = temperature
        if aggregator is not None:
            aggregator.seed = seed; aggregator.temp_override = temperature
        print(f"\n########## PASS {pi + 1}/{passes} (seed={seed}, temp={temperature}) ##########")
        all_stats, halted = _run_single_pass(datasets_by_category, vllm_client, model_name, num_frames,
                                             dataset, strategy, frame_budget, aggregator, summary_max_tokens,
                                             pass_tag=f"_pass{seed}", log_reasoning=log_reasoning)
        table = _format_table(all_stats, strategy, num_frames, header_extra=f" — pass {pi + 1} seed={seed}")
        print("\n" + table); _write_results_txt(model_name, dataset, table)
        per_pass_scalars.append({c: _category_scalar(s) for c, s in all_stats.items()})
        mc = [(s["correct"], s["total"]) for s in all_stats.values()
              if not s.get("rouge_scores") and not s.get("counting_scores")]
        per_pass_mc.append((sum(c for c, _ in mc), sum(t for _, t in mc)))
        if halted:
            print("[BUDGET] stopping passes early."); break

    seeds_used = [seeds[i] if i < len(seeds) else seeds[-1] + (i - len(seeds) + 1) for i in range(len(per_pass_scalars))]
    lines = ["=" * 60,
             f"ERROR BARS over {len(per_pass_scalars)} pass(es) ({strategy}, {num_frames} frames, "
             f"temp={temperature}, seeds={seeds_used}):", "=" * 60]
    for c in datasets_by_category.keys():
        vals = [p[c] for p in per_pass_scalars if c in p]
        if not vals:
            continue
        mean = statistics.mean(vals); std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(f"{c:20s}: {mean:.2%} +/- {std:.2%}  (n_pass={len(vals)})")
    mc_accs = [c / t for c, t in per_pass_mc if t]
    if mc_accs:
        lines.append("-" * 60)
        std = statistics.stdev(mc_accs) if len(mc_accs) > 1 else 0.0
        lines.append(f"{'MC Overall':20s}: {statistics.mean(mc_accs):.2%} +/- {std:.2%}")
    lines.append("=" * 60)
    table = "\n".join(lines)
    print("\n" + table); _write_results_txt(model_name, dataset, table)
