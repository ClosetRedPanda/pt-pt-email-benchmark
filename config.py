"""Minimal benchmark configuration.

The benchmark core is local/deterministic. Network access is used only when
calling candidate models through OpenRouter. LanguageTool is local via the
language_tool_python package.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_REFERENCE = DATA_DIR / "analysis_reference.jsonl"
ELABORATION_PROMPTS = DATA_DIR / "elaboration_prompts_pt_pt.json"
ELABORATION_CONSTRAINTS = DATA_DIR / "elaboration_constraints.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_COMPLETIONS_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

LANGUAGETOOL_LANG = "pt-PT"
LANGUAGETOOL_LANG_EN = "en-US"

DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 8

# Per-request HTTP timeout (seconds) used by OpenRouterClient.
DEFAULT_REQUEST_TIMEOUT = DEFAULT_TIMEOUT

# Adaptive rate limiting for OpenRouterClient: starting requests/sec and the
# floor/ceiling it is allowed to adapt between.
DEFAULT_INITIAL_RPS = 4.0
DEFAULT_MIN_RPS = 0.5
DEFAULT_MAX_RPS = 16.0

SYSTEM_PROMPT_ANALYSIS = """You are an expert multilingual email analysis engine.\nReturn ONLY valid JSON matching the supplied schema. Extract the requested fields faithfully from the email. For PT-PT emails, write free-text fields in natural European Portuguese; for English emails, write them in English. Do not invent entities or facts."""

SYSTEM_PROMPT_ELABORATION = """You are an executive email communication assistant. Draft a complete, ready-to-send professional email from the supplied task. Never use bracket placeholders. When the task asks for European Portuguese, use PT-PT vocabulary, grammar, orthography, and pronoun placement; avoid Brazilian Portuguese terms and gerund constructions."""

# Conservative score presentation: dimensions remain separate; there is no
# headline "overall quality" number.
#
# lean-1.1: the generation criteria were retargeted to close a permissive-pattern
# scoring floor (a content-free boilerplate reply used to earn 13.33% mean
# adherence, and up to 66.7% on a single task). That changes what an adherence
# score measures, so per docs/RELEASING.md the comparability identifier moves.
# No published artifact declared lean-1.0, so nothing is orphaned by this bump;
# from here, `tools/check_scoring_floor.py` keeps the floor pinned at zero.
BENCHMARK_VERSION = "lean-1.1"
