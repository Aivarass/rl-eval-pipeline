# LLM Eval Pipeline

Automated quality evaluation for RL agent discoveries, using LLM as Judge, rule based validation, and closed loop reward feedback.

Built to solve a real problem: an RL agent finds hundreds of potential API defects, but without evaluation most are noise. This pipeline separates genuine bugs from false positives, measures judge reliability against human labels, and feeds assessment signals back into the agent's reward function to improve future exploration.

Companion project to [SARSA REST Bug Hunter](https://github.com/Aivarass/autonomous-sarsa-rest-agent/tree/eval_pipeline_integration), a reinforcement learning agent that autonomously discovers multi step API bugs.

## The Problem

RL testing agents are good at finding patterns that produce 500 errors. They're bad at knowing which ones matter.

Without evaluation, the agent exploits a single high reward pattern and generates hundreds of redundant discoveries. In early runs, the bug hunter produced 600+ discoveries in 4 minutes, all variations of the same `GET_ALL → DELETE → 500` sequence. The agent had no signal telling it to explore new patterns instead of repeating known ones. This pipeline exists to close that gap, and feed evaluation signals back into the agent to shape what it explores next.

## Results

Three configurations were compared across 100K+ episode training runs on a target REST API with four resource chains (items → prices → discounts → points) and a hidden bug reachable only through a specific 5 step dependency sequence.

The agent autonomously discovered a bug requiring a 5 step dependency chain (Item → Price → Discount → Point → DELETE) with no domain knowledge, no schema, and no hardcoded exploration rules.

| Metric | Baseline (no eval) | Eval Pipeline (no decay) | Eval + Novelty Decay |
|--------|-------------------|--------------------------|----------------------|
| Unique bug combos | 47 | 57 | 154 |
| Endpoints explored | 1-2 | 2-3 | All 4 |
| Hidden bug discovered | No | No | Yes, episode 26,505 |

Without evaluation, the agent collapsed onto `DELETE /items` and repeated it indefinitely. With severity only evaluation, it explored slightly more but still gravitated toward the easiest high reward paths. With novelty decay, unique discoveries tripled and the agent was pushed deep enough into the state space to reach the final endpoint in the chain, something neither previous configuration achieved.

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
│  Severity classification     │  severity, category, root_cause,
│  Root cause analysis         │  novelty (vs prior findings)
│  Novelty assessment          │
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
│  Reward adjustment           │  base × novelty score,
│  Agent behaviour shaping     │  false positive → -1
└─────────────────────────────┘
```

## How It Works

### Feedback Loop and Novelty Scoring

This closed loop architecture, RL exploration shaped by LLM judgment, applies the same principle as RLHF to automated testing: using qualitative evaluation to guide quantitative optimisation.

Each discovery is sent to GPT-4o-mini with a structured prompt. The judge returns a validated JSON assessment:

```json
{
    "is_genuine_bug": true,
    "confidence": 0.9,
    "severity": "high",
    "category": "error_handling",
    "root_cause": "DELETE returns 500 after valid resource creation sequence",
    "novelty": 0.9
}
```

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
    base = severity_rewards.get(judge_result.severity, 10)
    return base * judge_result.novelty
```

The reward is the product of severity and novelty. The first discovery of a high severity bug returns `10 * 1.0 = 10`. The 20th rediscovery of the same root cause might return `10 * 0.05 = 0.5`. False positives are penalised at -1. The agent naturally shifts exploration toward unexplored patterns because known patterns stop paying.

As discoveries accumulate, the pipeline builds a compressed summary of known bug patterns from the evaluated batch:

```
Previously identified bug patterns:
- "DELETE returns 500 after valid resource creation sequence" (seen 12x, severity: high, category: error_handling)
- "Discount deletion fails with cascading reference error" (seen 3x, severity: high, category: data_integrity)
```

This summary is injected into the judge's prompt alongside each new discovery. The judge returns a `novelty` score (0.0 to 1.0) based on root cause similarity, not API sequence similarity. Different sequences that trigger the same underlying bug score low. A genuinely new root cause scores high.

The summary is grouped by root cause text from previous evaluations. Grouping happens at the semantic level (what the LLM identified as the root cause) rather than at the structural level (which API endpoints were called). The LLM performing the novelty assessment reads through wording variations, so even if the same bug gets slightly different root cause descriptions across evaluations, the novelty assessment still recognises the pattern as saturated.

### Design Journey

Initial runs with flat +10 reward collapsed into exploitation. The agent repeated DELETE /items 105,000 times per window because every hit paid the same. Adding the LLM judge with severity-based rewards was a clear improvement. The judge correctly filtered false positives, produced actionable bug reports with severity and root cause, and pushed unique discoveries from 47 to 57. But the exploitation trap remained because repeat combos bypassed the judge entirely and still received +10. Removing rewards for repeat combos caused the opposite failure, learned helplessness where the agent stopped executing altogether. The breakthrough was combining count-based decay for repeat discoveries with LLM novelty scoring for new ones. Known bugs pay less over time, genuinely new discoveries pay full reward based on the judge's severity and novelty assessment. This created natural exploration pressure without any hardcoded coverage rules. The result: 154 unique bug combos across all four endpoints, including a 5-level dependency chain discovery that no previous configuration achieved.

## LLM as Judge Evaluation

The judge is evaluated against a golden dataset of 45 human labelled examples, a mix of genuine bugs, noise, and non bugs, each manually classified with severity and category.

| Metric | Score |
|--------|-------|
| Accuracy | 97.8% |
| Precision | 93.3% |
| Recall | 100% |
| F1 | 96.6% |

The judge excels at binary bug detection (100% recall) but polarises severity classification toward extremes and defaults to `error_handling` for categories. These findings inform where human review is still needed.

**Judge reliability** is measured through two additional checks. Consistency measurement runs the same discovery through the judge multiple times and tracks agreement across bug detection, severity, category, and confidence spread. Confidence calibration checks whether the judge's stated confidence matches actual accuracy, when the judge says confidence=0.9, the Expected Calibration Error (ECE) measures whether it's actually correct 90% of the time.

## Pipeline Details

### Rule Based Checks

Rule checks run first as a fast pre filter before the expensive LLM layer. They validate input completeness, enforce type constraints, check API sequence dependency order, detect duplicates through sequence normalisation, and verify `final_status` consistency.

### Statistical Analysis

Statistical analysis operates on evaluated batches. Distribution summaries track bug rates, severity breakdowns, and category distributions. Batch comparison uses chi squared tests to detect distribution drift between runs. Outlier detection flags episode ranges with abnormally high false positive rates.

### Duplicate Detection

The RL agent naturally exploits high reward patterns, repeating variations like `GET_ALL → DELETE × 1` through `GET_ALL → DELETE × 12`. Sequence normalisation collapses consecutive repeated calls to the same key before hashing, so the first occurrence passes through and subsequent variations are filtered. This reduced discovery volume from 600+ to ~26 per run while preserving all unique findings.

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
    test_quality_report.py # Novelty summary building tests
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

- [SARSA REST Bug Hunter](https://github.com/Aivarass/autonomous-sarsa-rest-agent/tree/eval_pipeline_integration) Python RL agent that autonomously discovers multi step API bugs. The data source for this evaluation pipeline.
- [SARSA REST Bug Hunter (Java)](https://github.com/Aivarass/sarsa-rest-bug-hunter) Original Java implementation with pure neural network (no ML libraries). Discovers 5 step bug chains across 3.2 quintillion possible paths.
- [SARSA RSPS Agent](https://github.com/Aivarass/sarsa-rsps-agent) RL agent that learns combat strategies in a multiplayer game environment. Demonstrates RL applied beyond testing.
