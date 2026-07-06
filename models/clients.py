"""Model clients: a vision client (local vLLM or OpenAI) and a text-only aggregator client.

All prompting that touches the wire lives here so that harness code stays about *packaging*
(which frames, in what layout) and evaluation code stays about *scoring*.
"""
from openai import OpenAI
import base64
import cv2
import datetime
import json
import os

from models.cost import METER

OPENAI_MODELS = {"gpt-5.2", "gpt-5", "gpt-4o", "o3"}


class TextLLM:
    """Text-only chat wrapper for the decentralized aggregation pass (pass 2).

    Two backends: a local vLLM OpenAI-compatible server (base_url set — e.g. reuse the same
    served VLM in text-only mode, no cloud cost) or the real OpenAI API (base_url None, for
    gpt-* aggregators). Constructed lazily (only when --strategy decentralized) so importing
    this module never forces an OpenAI() that would fail without a key on disk.
    """

    def __init__(self, model, api_key=None, base_url=None):
        self.local = base_url is not None
        if self.local:
            self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self._tokens_key = "max_completion_tokens" if model in OPENAI_MODELS else "max_tokens"
        self.seed = None            # set per-pass for error bars
        self.temp_override = None   # set per-pass for sampling

    def prompt(self, p, max_tokens=None):
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": p}]}
        if not self.local:
            kwargs["store"] = False
        if max_tokens is not None:
            kwargs[self._tokens_key] = max_tokens
        if self._tokens_key == "max_tokens":  # local / non-reasoning models accept temperature
            kwargs["temperature"] = self.temp_override if self.temp_override is not None else 0.0
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = self.client.chat.completions.create(**kwargs)
        METER.record(self.model, getattr(response, "usage", None))
        return response.choices[0].message.content or ""


class VLLMClient:
    def __init__(self, api_base, model, dataset, strategy, num_frames, log_dir=None):
        if model in OPENAI_MODELS:
            self.client = OpenAI()
            self._tokens_key = "max_completion_tokens"
        else:
            self.client = OpenAI(api_key="EMPTY", base_url=api_base)
            self._tokens_key = "max_tokens"
        self.model = model
        self.seed = None            # set per-pass for error bars
        self.temp_override = None   # set per-pass for sampling
        print(f"Using model: {self.model}")

        if log_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(repo_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model.replace("/", "_")
        self._log_path = os.path.join(
            log_dir, f"log_{model_clean}_{dataset}_{strategy}_{num_frames}f_{timestamp}.json"
        )
        self._log_entries = []

    def _create(self, messages, tokens, temperature):
        kwargs = {"model": self.model, "messages": messages, self._tokens_key: tokens}
        if self._tokens_key == "max_tokens":  # local / non-reasoning: temperature allowed
            kwargs["temperature"] = self.temp_override if self.temp_override is not None else temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        resp = self.client.chat.completions.create(**kwargs)
        METER.record(self.model, getattr(resp, "usage", None))
        return resp

    def _log(self, user_content, max_tokens, temperature, chat_response):
        def strip_images(content):
            return [
                {"type": c["type"], "text": c.get("text", "<image>")} if c["type"] == "image_url" else c
                for c in content
            ]
        self._log_entries.append({
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "user_content": strip_images(user_content), "response": chat_response.model_dump(),
        })
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self._log_entries, f, indent=4, ensure_ascii=False)

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def sample_frames(self, frames_data, strategy_type="uniform", camera_names=None) -> list:
        user_content = []
        if strategy_type == "stitched":
            user_content.append({"type": "text", "text": "The following is a sequence of multi-camera grid images"})
            for encoded in [self._encode_frame(f) for f in frames_data]:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        else:
            frames_by_cam = frames_data
            if len(frames_by_cam) == 1:
                user_content.append({"type": "text", "text": "The following is the sequence of images"})
                for encoded in [self._encode_frame(f) for f in list(frames_by_cam.values())[0]]:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            else:
                user_content.append({"type": "text", "text": "The following is the sequence of images from multiple cameras"})
                for cam_name, frames in frames_by_cam.items():
                    display = (camera_names or {}).get(cam_name, cam_name)
                    user_content.append({"type": "text", "text": f"{display}:"})
                    for encoded in [self._encode_frame(f) for f in frames]:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return user_content

    def multiple_choice(self, frames_data, question, candidates, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = (
            "You must only return the letter of the answer choice, and nothing else. "
            "Do not include any other symbols, information, text, or justification in your answer. "
            "For example, if the correct answer is 'a) ...', you must only return 'a'."
        )
        prompt = f"{question}\n" + "".join(f"{c}\n" for c in candidates) + f"\n[PARSING RULE]: {parsing_rule}"
        user_content.append({"type": "text", "text": prompt})

        tokens = 5 if self.model in {"gpt-5.2", "gpt-5", "o3"} else 1
        chat_response = self._create([{"role": "user", "content": user_content}], tokens, 0.0)
        self._log(user_content, tokens, 0.0, chat_response)
        result = chat_response.choices[0].message.content.lower().strip()
        return result[0] if (tokens == 5 and result) else result

    def counting(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        parsing_rule = "You must only return a single number as your answer, and nothing else."
        user_content.append({"type": "text", "text": f"{question}\n\n[PARSING RULE]: {parsing_rule}"})
        chat_response = self._create([{"role": "user", "content": user_content}], 10, 0.0)
        self._log(user_content, 10, 0.0, chat_response)
        return chat_response.choices[0].message.content.strip()

    def summarize(self, frames_data, question, strategy_type="uniform", camera_names=None) -> str:
        user_content = self.sample_frames(frames_data, strategy_type, camera_names)
        user_content.append({"type": "text", "text": question})
        chat_response = self._create([{"role": "user", "content": user_content}], 1024, 0.7)
        self._log(user_content, 1024, 0.7, chat_response)
        return chat_response.choices[0].message.content.strip()

    def summarize_camera(self, frames, question, cam_label, max_tokens=512) -> str:
        """Query-conditioned summary of ONE camera feed (decentralized harness, pass 1).

        Reuses the single-camera path of sample_frames; the text-only aggregator (pass 2)
        later answers the question from these per-camera summaries without seeing the video.
        """
        user_content = self.sample_frames({cam_label: frames}, "uniform", {cam_label: cam_label})
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
        user_content.append({"type": "text", "text": instruction})
        chat_response = self._create([{"role": "user", "content": user_content}], max_tokens, 0.7)
        self._log(user_content, max_tokens, 0.7, chat_response)
        return chat_response.choices[0].message.content.strip()
