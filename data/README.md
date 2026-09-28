# Multi-Model Benchmark Dataset

This directory contains benchmark tasks across 4 evaluation domains:
1. `Reasoning`: Logical deduction, math word problems, and multi-step reasoning.
2. `Structured_Output`: Extraction into strict JSON schema representations.
3. `Multilingual`: Translation accuracy, idioms, and cross-lingual comprehension.
4. `Code_Gen`: Algorithm generation and refactoring tasks.

## Dataset Schema (`comparison_benchmark_dataset.jsonl`)

```json
{
  "task_id": "bm_001",
  "domain": "Reasoning",
  "prompt": "If 5 workers build 5 tables in 5 days, how many days for 100 workers to build 100 tables?",
  "reference_answer": "It takes 5 days because 1 worker builds 1 table in 5 days.",
  "require_json": false,
  "model_outputs": {
    "gpt-4o": {"output": "Therefore, 1 worker builds 1 table in 5 days. Thus, 100 workers build 100 tables in 5 days.", "latency_ms": 420, "input_tokens": 100, "output_tokens": 50},
    "claude-3-5-sonnet": {"output": "Since 1 worker takes 5 days per table, 100 workers will take 5 days to build 100 tables.", "latency_ms": 390, "input_tokens": 100, "output_tokens": 45}
  }
}
```
