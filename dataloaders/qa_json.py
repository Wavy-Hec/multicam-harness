"""Loader for CrossView-style QA JSONs: one qa_<category>.json per category in a directory.

Datasets covered: meva (CrossView/MEVA), nuscenes, ego-exo4d, agibot — anything emitted by the
CrossView dataset-construction pipeline. Each entry carries question/answer/options plus
video_paths + camera_names, and optional stsg_metadata greybox fields (which cameras are
actually needed to answer).
"""
import json
import os

COUNTING_CATS = {"counting"}
SUMMARIZATION_CATS = {"summarization"}

_DEFAULT_CATEGORIES = {
    "nuscenes": ["counting", "spatial", "temporal", "event_ordering", "summarization"],
    "meva": ["counting", "spatial", "temporal", "event_ordering", "best_camera", "summarization"],
    "ego-exo4d": ["temporal", "event_ordering", "best_camera", "summarization"],
    "agibot": ["temporal", "event_ordering", "summarization"],
}


def categories_for(dataset):
    """Category list for a dataset — from configs/datasets.yaml if present, else the defaults."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, "configs", "datasets.yaml")
    try:
        import yaml
        with open(cfg_path) as f:
            dist = yaml.safe_load(f).get("categories", {})
        if dataset in dist:
            return list(dist[dataset])
    except Exception:
        pass
    return _DEFAULT_CATEGORIES[dataset]


def load_datasets(dataset, data_dir):
    categories = categories_for(dataset)
    datasets_by_category = {}
    for category in categories:
        file_path = os.path.join(data_dir, f"qa_{category}.json")
        if not os.path.exists(file_path):
            print(f"  (skip missing {file_path})")
            continue
        with open(file_path) as f:
            data = json.load(f)

        category_dict = {}
        for i, entry in enumerate(data):
            if not all(k in entry for k in ("question", "answer", "video_paths")):
                continue
            scene = entry.get("scene_id", "scene")
            qtype = entry.get("question_type", category)
            key = f"{scene}_{qtype}_{i}"
            options = entry.get("options")
            if options is not None:
                # Normalize dict-form options ({"A": "Camera G424", ...}) to the
                # list form the MC prompt expects (["A. Camera G424", ...]). The
                # organized-dir QA JSONs use the dict form; iterating a dict would
                # otherwise drop the option text and leave only the letters.
                if isinstance(options, dict):
                    options = [f"{k}. {v}" for k, v in sorted(options.items())]
                candidates = options
                correct_answer = entry["answer"].lower()
            else:
                candidates = None
                correct_answer = entry["answer"]
            sm = entry.get("stsg_metadata", {})
            category_dict[key] = {
                "question": entry["question"],
                "candidates": candidates,
                "correct_answer": correct_answer,
                "question_type": qtype,
                "video_paths": entry.get("video_paths", []),
                "camera_names": entry.get("camera_names", {}),
                # ground-truth greybox fields: which cameras are actually needed + why
                "needed_cameras": sm.get("cameras"),
                "num_needed_cameras": sm.get("num_cameras"),
                "gt_reasoning": entry.get("reasoning"),
            }
        print(f"Using {len(category_dict)} {category} questions")
        datasets_by_category[category] = category_dict
    return datasets_by_category


def shard_datasets(datasets_by_category, shard):
    """'i:N' — keep only questions whose index mod N == i within each category (sorted keys).
    Enables running N parallel processes over disjoint question slices."""
    si, sn = (int(x) for x in shard.split(":"))
    for cat, d in datasets_by_category.items():
        keys = sorted(d.keys())
        datasets_by_category[cat] = {k: d[k] for idx, k in enumerate(keys) if idx % sn == si}
    print(f"[SHARD] {si}:{sn} -> " + ", ".join(f"{c}={len(d)}" for c, d in datasets_by_category.items()))
    return datasets_by_category
