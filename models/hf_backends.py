"""In-process HF-transformers backends (Qwen3-VL / InternVL3) for GPU boxes with no served
vLLM endpoint (e.g. SLURM). Ported from Hector's hector-port branch (models/clients.py there),
with two changes:

  - Stripped of that branch's own video-sampling path (load_video/get_index): master's harnesses
    already produce decoded frame lists (uniform_sampling_strategy, stitched_frames_sampling_strategy),
    so these backends only need "N already-decoded frames + text -> text out," not their own
    decord pipeline. Every image is passed as a {"type": "image"} chat-content entry.
  - Fixed a token-double-count bug in InternVL3Backend.generate: the ported version tokenized
    the full prompt (which already contains literal "<image>\\n" placeholder text) as ordinary
    text tokens, THEN added img_tokens on top for the same images. Placeholders are stripped
    before counting text tokens here, so input_tokens is (real text) + (real image expansion),
    comparable to Qwen's <|video_pad|> count.

HFClient (bottom of file) is the adapter other code actually talks to: it exposes the same
multiple_choice/counting/summarize/summarize_camera surface as models.clients.VLLMClient, so
runner.py and the harnesses (stitched/decentralized/uniform) need zero changes to use either
backend — only run_vqa.py's client-construction site picks between them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from models.cost import METER


@dataclass
class GenOut:
    """One VLM generation call's output + accounting."""
    text: str
    input_tokens: int
    video_tokens: int
    output_tokens: int
    latency_s: float


class Backend:
    """A VLM that turns chat ``content`` (a list of text/image parts) into text + stats."""
    name = "backend"

    def generate(self, content, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        raise NotImplementedError


def load_model(model_path, dtype):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    return model, processor


# ---------------------------------------------------------------------------
# Qwen backend
# ---------------------------------------------------------------------------

class QwenBackend(Backend):
    def __init__(self, model_path, dtype="bfloat16"):
        self.model_path = model_path
        self.name = model_path.rstrip("/").split("/")[-1]
        dt = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model, self.processor = load_model(model_path, dt)
        # Qwen3-VL uses 16px patches, Qwen2/2.5-VL 14px.
        self.patch_size = getattr(self.processor.image_processor, "patch_size", None) or 14
        from qwen_vl_utils import process_vision_info
        self._pvi = process_vision_info
        self._vid_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def generate(self, content, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        messages = [{"role": "user", "content": content}]
        proc = self.processor
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self._pvi(
            messages, return_video_kwargs=True, return_video_metadata=True,
            image_patch_size=self.patch_size)
        if video_inputs:  # only unzip a genuinely non-empty list (not just non-None)
            video_inputs, video_metadata = map(list, zip(*video_inputs))
        else:
            video_inputs, video_metadata = None, None
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      video_metadata=video_metadata, do_resize=False, padding=True,
                      return_tensors="pt", **video_kwargs).to(self.model.device)
        total = int(inputs.input_ids.shape[1])
        vid = int((inputs.input_ids[0] == self._vid_id).sum())
        do_sample = temperature is not None and temperature > 0
        if do_sample and seed is not None:
            torch.manual_seed(seed)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self.model.generate(**inputs, **gen_kwargs)
        dt = time.perf_counter() - t0
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = proc.batch_decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
        return GenOut(text=out, input_tokens=total, video_tokens=vid,
                      output_tokens=int(trimmed.shape[1]), latency_s=dt)


QWEN_ALIASES = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3vl-instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}


# ---------------------------------------------------------------------------
# InternVL3 backend
#
# IMPORTANT: run this backend under a conda env with transformers 4.48.3 — newer
# transformers breaks the InternVL3 remote code.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(im) for im in images])


class InternVL3Backend(Backend):
    def __init__(self, model_path="OpenGVLab/InternVL3-8B", max_tiles=1, device="cuda:0"):
        from transformers import AutoModel, AutoTokenizer
        self.model_path = model_path
        self.name = model_path.rstrip("/").split("/")[-1]
        self.max_tiles = max_tiles
        self.device = device

        # InternVL remote code calls .item() on a torch.linspace during __init__;
        # route device-less linspace to CPU while from_pretrained constructs the model.
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        torch.linspace = _cpu_linspace
        try:
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, device_map=device).eval()
        finally:
            torch.linspace = _orig_linspace

        # Without flash-attn the remote code forces eager attention, which OOMs on
        # multi-image prompts; the LLM picks its kernel from config at runtime.
        lm_cfg = getattr(getattr(self.model, "language_model", None), "config", None)
        if lm_cfg is not None and getattr(lm_cfg, "_attn_implementation", None) == "eager":
            lm_cfg._attn_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.num_image_token = getattr(self.model, "num_image_token", 256)

    def _flatten(self, content):
        """content -> (question_str, pixel_values|None, num_patches_list|None).

        Every image entry is one already-decoded frame; there is no {"type": "video"}
        branch here (unlike the ported version) — frame sampling is the harness's job,
        this backend only tiles/embeds whatever images it's handed.
        """
        parts, pixel_chunks, npl = [], [], []
        for item in content:
            t = item.get("type")
            if t == "text":
                parts.append(item["text"])
            elif t == "image":
                px = load_image(item["image"], input_size=448, max_num=self.max_tiles)
                pixel_chunks.append(px)
                npl.append(px.shape[0])
                parts.append("<image>\n")
        question = "".join(parts)
        if pixel_chunks:
            pixel_values = torch.cat(pixel_chunks, dim=0).to(torch.bfloat16).to(self.device)
            return question, pixel_values, npl
        return question, None, None

    def generate(self, content, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        question, pixel_values, npl = self._flatten(content)
        do_sample = temperature is not None and temperature > 0
        if do_sample and seed is not None:
            torch.manual_seed(seed)
        gen_cfg = dict(num_beams=1, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_cfg.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, question, gen_cfg,
                                       num_patches_list=npl, history=None, return_history=True)[0]
        dt = time.perf_counter() - t0

        # Token accounting: strip the "<image>\n" placeholders before tokenizing text,
        # so image cost is counted once (via img_tokens), not twice.
        text_only = question.replace("<image>\n", "")
        text_tokens = (int(self.tokenizer(text_only, return_tensors="pt").input_ids.shape[1])
                      if text_only else 0)
        img_tokens = self.num_image_token * (sum(npl) if npl else 0)
        out_tokens = int(self.tokenizer(response, return_tensors="pt").input_ids.shape[1])
        return GenOut(text=response, input_tokens=text_tokens + img_tokens,
                      video_tokens=img_tokens, output_tokens=out_tokens, latency_s=dt)


INTERNVL_ALIASES = {"internvl3": "OpenGVLab/InternVL3-8B"}


def make_backend(alias, max_tiles=1, device="cuda:0"):
    if alias in QWEN_ALIASES:
        return QwenBackend(QWEN_ALIASES[alias])
    if alias in INTERNVL_ALIASES:
        return InternVL3Backend(INTERNVL_ALIASES[alias], max_tiles=max_tiles, device=device)
    if "internvl" in alias.lower():
        return InternVL3Backend(alias, max_tiles=max_tiles, device=device)
    return QwenBackend(alias)


# ---------------------------------------------------------------------------
# HFClient — the adapter runner.py / harnesses actually call
# ---------------------------------------------------------------------------

class _UsageShim:
    """Minimal object satisfying CostMeter.record's getattr(usage, "prompt_tokens", ...)."""

    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class HFClient:
    """Exposes VLLMClient's/TextLLM's call surface over an in-process HF backend, so
    runner.py, decentralized.py, and stitched.py need no changes to use it."""

    def __init__(self, model, device="cuda:0", max_tiles=1):
        self.model = model
        self.backend = make_backend(model, max_tiles=max_tiles, device=device)
        self.seed = None            # set per-pass for error bars
        self.temp_override = None   # set per-pass for sampling

    def _encode_frame(self, frame):
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def sample_frames(self, frames_data, strategy_type="uniform", camera_names=None) -> list:
        content = []
        if strategy_type == "stitched":
            content.append({"type": "text", "text": "The following is a sequence of multi-camera grid images"})
            for f in frames_data:
                content.append({"type": "image", "image": self._encode_frame(f)})
        else:
            frames_by_cam = frames_data
            if len(frames_by_cam) == 1:
                content.append({"type": "text", "text": "The following is the sequence of images"})
                for f in list(frames_by_cam.values())[0]:
                    content.append({"type": "image", "image": self._encode_frame(f)})
            else:
                content.append({"type": "text", "text": "The following is the sequence of images from multiple cameras"})
                for cam_name, frames in frames_by_cam.items():
                    display = (camera_names or {}).get(cam_name, cam_name)
                    content.append({"type": "text", "text": f"{display}:"})
                    for f in frames:
                        content.append({"type": "image", "image": self._encode_frame(f)})
        return content

    def _generate(self, content, max_new_tokens, temperature):
        temp = self.temp_override if self.temp_override is not None else temperature
        out = self.backend.generate(content, max_new_tokens, seed=self.seed, temperature=temp)
        METER.record(self.model, _UsageShim(out.input_tokens, out.output_tokens))
        return out

    def multiple_choice(self, frames_data, question, candidates, strategy_type="uniform", camera_names=None) -> str:
        content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = (
            "You must only return the letter of the answer choice, and nothing else. "
            "Do not include any other symbols, information, text, or justification in your answer. "
            "For example, if the correct answer is 'a) ...', you must only return 'a'."
        )
        prompt = f"{question}\n" + "".join(f"{c}\n" for c in candidates) + f"\n[PARSING RULE]: {parsing_rule}"
        content.append({"type": "text", "text": prompt})
        out = self._generate(content, 8, 0.0)
        result = out.text.strip().lower()
        return result[0] if result else result

    def counting(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = "You must only return a single number as your answer, and nothing else."
        content.append({"type": "text", "text": f"{question}\n\n[PARSING RULE]: {parsing_rule}"})
        out = self._generate(content, 10, 0.0)
        return out.text.strip()

    def summarize(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        content = self.sample_frames(frames_data, strategy_type, camera_names)
        content.append({"type": "text", "text": question})
        out = self._generate(content, 1024, 0.7)
        return out.text.strip()

    def summarize_camera(self, frames, question, cam_label, max_tokens=512) -> str:
        content = self.sample_frames({cam_label: frames}, "uniform", {cam_label: cam_label})
        instruction = (
            f"You are analyzing a SINGLE camera feed labeled '{cam_label}'. "
            "A question will later be answered only from your written description — you will NOT "
            f"see this footage again. The question is:\n\"{question}\"\n\n"
            "Describe everything in THIS feed that is relevant to answering it: the people and "
            "objects present and their counts, their actions and interactions, the temporal order "
            "in which events happen, and whether the queried event or entity appears here. Be "
            "concrete and specific (state counts, directions of movement, and ordering). If nothing "
            "in this feed is relevant to the question, say so explicitly."
        )
        content.append({"type": "text", "text": instruction})
        out = self._generate(content, max_tokens, 0.7)
        return out.text.strip()
