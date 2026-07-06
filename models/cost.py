"""Token/cost accounting shared by every model call in a run."""
import os


class BudgetExceeded(Exception):
    """Raised to stop a run once the --max_usd cap is hit."""


class CostMeter:
    """Accumulates token usage and (for priced models) USD cost across every model call in a run.
    Shared by VLLMClient and TextLLM so decentralized's per-camera + aggregation calls all count.
    Local vLLM models have no price entry -> cost stays 0 but tokens are still logged."""

    def __init__(self):
        self.prices = {}      # {model: {"input": usd_per_1M, "output": usd_per_1M}}
        self.max_usd = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usd = 0.0
        self.calls = 0

    def configure(self, prices=None, max_usd=None):
        self.prices = prices or {}
        self.max_usd = max_usd

    def record(self, model, usage):
        self.calls += 1
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        p = self.prices.get(model)
        if p:
            self.usd += pt / 1e6 * p.get("input", 0.0) + ct / 1e6 * p.get("output", 0.0)

    def over_cap(self):
        return self.max_usd is not None and self.usd >= self.max_usd


# One meter per process: VLLMClient and TextLLM both record into it, so the
# decentralized harness's per-camera + aggregation calls are counted together.
METER = CostMeter()


def load_prices(path):
    if not path:
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"(no price table at {path}: {e})")
        return {}


def estimate_cost(datasets_by_category, strategy, num_frames, model, prices):
    """Order-of-magnitude pre-run USD estimate for a priced (OpenAI) model. Image-token counts vary
    a lot, so this is a heads-up, not a guarantee — the live --max_usd cap is the real guard.
    Returns (usd, n_questions) or (None, n) when the model isn't priced (local vLLM)."""
    n = sum(len(d) for d in datasets_by_category.values())
    p = prices.get(model)
    if not p:
        return None, n
    IMG_TOK, TXT_IN = 800, 400  # rough tokens per frame / per text prompt
    if strategy == "decentralized":
        avg_cams = 3  # true value is per-question; ~3 is typical for the gauge subset
        in_tok = avg_cams * (num_frames * IMG_TOK + TXT_IN) + (avg_cams * 512 + TXT_IN)
        out_tok = avg_cams * 512 + 16
    else:  # uniform / stitched: one VLM call per question
        in_tok = num_frames * IMG_TOK + TXT_IN
        out_tok = 64
    usd = n * (in_tok / 1e6 * p.get("input", 0.0) + out_tok / 1e6 * p.get("output", 0.0))
    return usd, n


def resolve_openai_key():
    """Find an OpenAI key. Our .env files use OPENAI_API_KEY1/OPENAI_API_KEY2 (numbered),
    not the standard OPENAI_API_KEY, so the default OpenAI() lookup would miss it."""
    key_vars = ("OPENAI_API_KEY1", "OPENAI_API_KEY2", "OPENAI_API_KEY")
    for var in key_vars:
        if os.environ.get(var):
            return os.environ[var]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, ".env"),         # multicam-harness/.env
        os.path.join(repo_root, "..", ".env"),    # parent workspace .env
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in candidates:
        if not os.path.exists(env_path):
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                for var in key_vars:
                    if line.startswith(var + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
