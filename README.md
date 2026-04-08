# LLM Eval Pipeline

Automated quality evaluation for RL agent discoveries, using LLM as Judge, rule based validation, and closed loop reward feedback.

Built to solve a real problem: an RL agent finds hundreds of potential API defects, but without evaluation most are noise. This pipeline separates genuine bugs from false positives, measures judge reliability against human labels, and feeds assessment signals back into the agent's reward function to improve future exploration.

Companion project to [SARSA REST Bug Hunter](https://github.com/Aivarass/autonomous-sarsa-rest-agent), a reinforcement learning agent that autonomously discovers multi step API bugs.

## The Problem

RL testing agents are good at finding patterns that produce 500 errors. They're bad at knowing which ones matter.

Without evaluation, the agent exploits a single high reward pattern and generates hundreds of redundant discoveries. In early runs, the bug hunter produced 600+ discoveries in 4 minutes, all variations of the same `GET_ALL → DELETE → 500` sequence. The agent had no signal telling it to explore new patterns instead of repeating known ones.

This pipeline exists to close that gap.

## Architecture

```
SARSA Bug Hunter (RL Agent)
    │
    │ raw discoveries (JSON)
    ▼
┌─────────────────────────────┐
│  Rule Based Checks           │  Completeness, type validation,
│  Input validation            │  dependency order, sequence
│  Duplicate detection         │  normalisation, duplicate filtering
│  Dependency validation       │
└─────────────┬───────────────┘
              │ valid discoveries
              ▼
┌─────────────────────────────┐
│  LLM Judge (GPT-4o-mini)     │  Structured evaluation:
│  Bug vs false positive       │  is_genuine_bug, confidence,
│  Severity classification     │  severity, category, root_cause
│  Root cause analysis         │
└─────────────┬───────────────┘
              │ evaluations
              ▼
┌─────────────────────────────┐
│  Statistical Analysis        │  Distribution summary,
│  Distribution tracking       │  batch comparison (chi squared),
│  Batch drift detection       │  outlier session detection
│  Outlier flagging            │
└─────────────┬───────────────┘
              │ quality signals
              ▼
┌─────────────────────────────┐
│  Feedback Loop               │  Reward modification:
│  Reward adjustment           │  genuine high → +10,
│  Agent behaviour shaping     │  genuine low → +3,
│                              │  false positive → -1
└─────────────────────────────┘
```

## How It Works

### LLM Integration

Each discovery from the bug hunter is sent to GPT-4o-mini with a structured prompt. The judge returns a validated JSON assessment:

```json
{
    "is_genuine_bug": true,
    "confidence": 0.9,
    "severity": "high",
    "category": "error_handling",
    "root_cause": "DELETE returns 500 after valid resource creation sequence"
}
```

Responses are validated through a `JudgeResult` dataclass that enforces field types, allowed values, and confidence bounds. Malformed LLM responses are caught before they enter the pipeline.

The model, temperature, and system prompt are all configurable through `judge_config.json`, so you can iterate on prompts or swap models without touching code.

### LLM as Judge Evaluation

The judge itself is evaluated against a golden dataset of 45 human labelled examples, a mix of genuine bugs, noise, and non bugs, each manually classified with severity and category.

**Results against golden dataset:**

| Metric | Score |
|--------|-------|
| Accuracy | 97.8% |
| Precision | 93.3% |
| Recall | 100% |
| F1 | 96.6% |

**Findings from failure mode analysis:**

The judge is excellent at binary bug detection. It catches every genuine bug and rarely flags noise as real. Severity classification is polarised: it reliably identifies high and low severity but struggles with medium and critical, defaulting to the extremes. Category classification is the weakest dimension at 49% accuracy. The judge defaults to `error_handling` for most findings and struggles to distinguish `state_management` and `data_integrity` from generic error patterns.

These findings directly inform where human review is still needed and where the judge can be trusted to operate autonomously.

### Judge Reliability

Two additional checks measure how trustworthy the judge actually is:

**Consistency measurement** runs the same discovery through the judge multiple times and tracks how often the answers agree. If the judge says "high severity bug" on one run and "low severity noise" on the next, that's a problem. The measurement covers bug detection agreement, severity agreement, category agreement, and confidence spread across runs.

**Confidence calibration** checks whether the judge's confidence scores mean what they say. When the judge says confidence=0.9, is it actually correct 90% of the time? Results are binned by confidence level and compared against actual accuracy. The Expected Calibration Error (ECE) gives a single number for how well calibrated the judge is. A well calibrated judge at 0.9 confidence should be right about 90% of the time; an overconfident one might only be right 50% of the time despite claiming 90%.

### Automated QA Pipeline

Three validation layers operate in sequence:

**Rule based checks** run first and are fast. They validate input completeness, enforce type constraints, check API sequence dependency order (items before prices before discounts before points), detect duplicates through sequence normalisation that collapses consecutive repeated calls, and verify `final_status` consistency.

**LLM judge** runs on discoveries that pass rule checks. This is the expensive layer, so garbage is filtered before it reaches the API.

**Statistical analysis** operates on evaluated batches. Distribution summaries track bug rates, severity breakdowns, and category distributions. Batch comparison uses chi squared tests via scipy to detect distribution drift between runs. Outlier detection flags episode ranges with abnormally high false positive rates.

The pipeline orchestrator (`QualityReport`) coordinates all three layers and supports both batch processing and real time single discovery evaluation.

### Feedback Loop

Evaluation results feed directly into the bug hunter's reward function:

```python
def adjust_reward(self, judge_result):
    if not judge_result.is_genuine_bug:
        return -1

    severity_rewards = {
        'low': 3,
        'medium': 6,
        'high': 10
    }
    return severity_rewards.get(judge_result.severity, 10)
```

Instead of a flat +10 for any 500 response, the agent's reward is now informed by the LLM judge's assessment. Genuine high severity bugs receive full reward. Low severity findings receive reduced reward. False positives are penalised. The agent learns what "good" means and adjusts its exploration accordingly.

**Why this matters:** Without feedback, the agent found one pattern and exploited it 600+ times. With feedback, the agent is incentivised to discover diverse, high quality bugs rather than repeating known patterns. This is the same closed loop architecture used in RLHF pipelines for model post training, where evaluation signals shape agent behaviour.

## Duplicate Detection and Sequence Normalisation

A key finding during development: the RL agent naturally exploits high reward patterns. Once it discovers that `GET_ALL → DELETE → 500` yields +10 reward, it repeats variations endlessly. `GET_ALL → DELETE × 1`, `GET_ALL → DELETE × 2`, up to `GET_ALL → DELETE × 12`. Each is technically a unique API sequence but semantically identical.

The pipeline handles this at two levels:

**In the bug hunter:** Only genuinely new bug combos emit discoveries. The agent still receives rewards for learning (SARSA updates continue) but duplicate patterns don't generate evaluation overhead.

**In the rule checks:** Sequence normalisation collapses consecutive repeated calls before hashing. `GET_ALL → DELETE × 12` normalises to the same key as `GET_ALL → DELETE × 1`. The first occurrence passes through; subsequent variations are filtered.

This reduced discovery volume from 600+ to ~26 per run while preserving all unique findings.

## Configuration

The judge is configured through `judge_config.json` at the project root:

```json
{
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "system_prompt": "You are an expert QA engineer..."
}
```

Settings are loaded with a three level priority: defaults in code, then overrides from the JSON config file, then any values passed directly in the constructor. This lets you iterate on prompts, swap models, or tune temperature without changing any Python code.

## Project Structure

```
eval_pipeline/
    config.py             # JudgeConfig dataclass, settings loading
    llm_judge.py          # LLM API integration and prompt logic
    judge_result.py       # JudgeResult dataclass with response validation
    judge_eval.py         # Judge consistency and confidence calibration
    rule_check.py         # Input validation, dependency checks, duplicate detection
    stat_checks.py        # Distribution analysis, batch comparison, outlier detection
    quality_report.py     # Pipeline orchestrator (batch and single discovery modes)
    eval_runner.py        # Golden dataset evaluation with metrics computation
data/
    golden/               # 45 human labelled examples for judge evaluation
    bugs/                 # Test data: genuine bug discoveries
    noise/                # Test data: noisy/ambiguous discoveries
    nonbugs/              # Test data: expected behaviour (not bugs)
tests/
    test_judge.py         # Basic LLM judge smoke test
    test_llm_judge.py     # LLM judge tests across bugs, noise, and non bugs
    test_eval_runner.py   # Golden dataset evaluation and metrics tests
    test_judge_result.py  # Response parsing and validation tests
    test_judge_eval.py    # Calibration and consistency tests
    test_rule_check.py    # Input, dependency, and duplicate detection tests
    test_stat_checks.py   # Distribution and outlier detection tests
judge_config.json         # External judge configuration
```

## Running

```bash
# Install dependencies
pip install -e .

# Set API key (or add to .env file)
export OPENAI_API_KEY=your_key_here

# Run the golden dataset evaluation
python -m eval_pipeline.eval_runner

# Run tests (offline tests work without an API key)
pytest tests/

# Run only offline tests
pytest tests/test_judge_result.py tests/test_rule_check.py tests/test_stat_checks.py tests/test_judge_eval.py
```

### End to end with the bug hunter

Start the target REST API, then run the SARSA agent. The agent discovers bugs, the eval pipeline evaluates each discovery in real time, and the feedback loop adjusts rewards:

```bash
# Terminal 1: Start the target API
mvn spring-boot:run

# Terminal 2: Run the agent with evaluation
python sarsa/SarsaRestTester.py
```

## Technologies

Python, OpenAI API (GPT-4o-mini), pytest, scipy (chi squared tests), SARSA (reinforcement learning)

## Related Projects

- [SARSA REST Bug Hunter](https://github.com/Aivarass/autonomous-sarsa-rest-agent) Python RL agent that autonomously discovers multi step API bugs. The data source for this evaluation pipeline.
- [SARSA REST Bug Hunter (Java)](https://github.com/Aivarass/sarsa-rest-bug-hunter) Original Java implementation with pure neural network (no ML libraries). Discovers 5 step bug chains across 3.2 quintillion possible paths.
- [SARSA RSPS Agent](https://github.com/Aivarass/sarsa-rsps-agent) RL agent that learns combat strategies in a multiplayer game environment. Demonstrates RL applied beyond testing.
