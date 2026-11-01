# Evaluation and benchmarks

These results come from real local runs on a small fictional corpus: 12 documents and 25
labeled questions. They demonstrate the method. They do not predict results on other
documents. Reports are saved to the ignored `artifacts/` directory.

```bash
uv run --locked python scripts/evaluate.py --mode retrieval
uv run --locked python scripts/evaluate.py --mode rag
uv run --locked python scripts/evaluate.py --report REPORT.json --judgments JUDGMENTS.json
uv run --locked python scripts/benchmark.py --points 10000 --queries 100 --seed 42 --top-k 10
```

## Dataset

`data/evaluation/questions.jsonl` holds 10 straightforward, 5 paraphrased, 5 multi-passage, and 5
unanswerable questions. Answerable questions carry a reference answer, relevant documents, and
quoted supporting passages. `corpus_lock.json` records the corpus fingerprint.

Before any run, the labels are checked against the documents. If the corpus changed, a source
is unknown, or a quote is missing, evaluation stops, because stale labels produce numbers that
look fine and mean nothing. Questions were written from the documents before results were seen.

## Retrieval metrics

The unit is the deduplicated document, within the top 5 chunks retrieved. Relevance labels are
exhaustive, so recall is meaningful. Unanswerable questions leave every denominator.

- **Hit Rate@K:** share of questions with any relevant document in the top K.
- **Recall@K:** relevant documents found over all relevant documents, averaged.
- **MRR@K:** reciprocal rank of the first relevant document, with zero for a miss, averaged.

Passage-level recall is N/A, because passage labels are illustrative, not exhaustive.

| Chunk / overlap | Hit Rate@5 | Recall@5 | MRR@5 | Top document | Mean latency |
|---|---|---|---|---|---|
| **256 / 40 (default)** | **100%** | **100%** | 0.912 | 85% | 6.6 ms |
| 128 / 24 | 95% | 95% | 0.950 | 95% | 6.7 ms |
| 64 / 12 | 95% | 95% | 0.900 | 85% | 6.9 ms |

Straightforward and multi-passage questions were perfect at every size. All differences come
from the five paraphrased questions:

| Chunk / overlap | Paraphrased Hit Rate | Paraphrased MRR |
|---|---|---|
| 256 / 40 | 100% | 0.650 |
| 128 / 24 | 80% | 0.800 |
| 64 / 12 | 80% | 0.600 |

**Decision: keep 256/40.** Smaller chunks put more relevant documents first, but “How much vacation
time do I get?” fell out of the top 5 at both smaller sizes. At 128/24 that caused the assistant
to refuse a question it could answer. The model can use a relevant passage at rank 5, but not one
that was never retrieved. This rests on one question out of 20, so it is a reason to prefer the
default here, not evidence about chunking in general. The real weakness is vocabulary: the policy
says “annual leave” and never “vacation”. Hybrid search or query expansion are possible
improvements, and neither was tested.

## RAG metrics

Run at 256/40, Top-5, temperature 0, with `qwen3:4b-instruct-2507-q4_K_M`.

| Metric | Result |
|---|---|
| Citation validity | 26 of 26 emitted references (100%) |
| Uncited substantive answers | 0 of 20 |
| Correct refusals, unanswerable questions | 5 of 5 |
| Incorrect refusals, answerable questions | 0 of 20 |
| Multi-passage evidence retrieved and retained | 5 of 5 |
| Generation latency | Mean 1,992 ms, p95 2,537 ms |

At 128/24, one answerable question was wrongly refused (1 of 20), caused by the retrieval miss.

### Correctness

Correctness is judged by a reviewer, not by string comparison, using this rubric: 0 is unsupported
or wrong, 1 is partially supported or missing part of the reference, 2 is fully supported.
Without judgments the metric is N/A. Reports name the reviewer.

The current scores come from the **implementing agent, not a human**, and are not an independent
evaluation.

| Score | Count |
|---|---|
| 2, fully supported | 20 of 25 |
| 1, partially supported | 5 of 25 |
| 0, unsupported or wrong | 0 of 25 |
| Mean | 1.8 of 2 |

The five partial scores: three omitted a condition or detail in the reference, one answered with
only “IT”, and one was correct but did not cite the document its claim came from. **That last
case had valid citations, which shows why validity is not correctness:** a metric of 100% still
missed a claim its cited source did not support.

Failures are assigned to the earliest stage: retrieval, context omission, or generation. All five
partial answers were generation issues, since the required evidence reached the model.

## Vector benchmark

Exact search versus ANN-permitted search on an indexed collection. See
[vector indexing](vector_indexing.md#benchmark-results) for the method and limits.

| Points | Mode | Recall@10 | Mean | p95 |
|---|---|---|---|---|
| 10,000 | exact | 1.000 | 2.18 ms | 3.33 ms |
| 10,000 | `hnsw_ef` 32 | 0.296 | 2.12 ms | 3.08 ms |
| 10,000 | `hnsw_ef` 256 | 0.867 | 2.69 ms | 3.66 ms |
| 50,000 | exact | 1.000 | 3.89 ms | 5.07 ms |
| 50,000 | `hnsw_ef` 32 | 0.085 | 2.15 ms | 3.05 ms |
| 50,000 | `hnsw_ef` 256 | 0.412 | 3.89 ms | 4.78 ms |

## Limits

- Twenty-five questions is a small sample: one question moves a percentage by 5 points.
- The questions and documents share an author, which favors retrieval.
- Latency comes from a single run on one Apple M4 Pro laptop.
- Refusal results cover five questions and do not establish a rate.
