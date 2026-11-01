"""Evaluate retrieval or RAG against the labeled questions, and save a real report.

  --mode retrieval                  needs Qdrant and the embedding model
  --mode rag                        also needs Ollama; writes a judgment template
  --report R --judgments J          aggregates reviewed scores without new generation

Exit codes: 0 report produced, 2 evaluation could not run.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_pipeline.assets import embedding_assets_id, resolve_embedding_tokenizer
from rag_pipeline.cli import run_cli
from rag_pipeline.config import Settings
from rag_pipeline.errors import EvaluationError, PipelineError
from rag_pipeline.evaluation.dataset import (
    corpus_fingerprint,
    load_questions,
    read_corpus_lock,
    validate_labels,
)
from rag_pipeline.evaluation.rag_metrics import (
    evaluate_rag,
    rows_from_json,
    rows_to_json,
    summarize,
    validate_judgments,
)
from rag_pipeline.evaluation.retrieval_metrics import evaluate_retrieval
from rag_pipeline.ingestion.ingestion_pipeline import prepare_documents, processing_fingerprint
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.rag.rag_service import load_rag_service
from rag_pipeline.retrieval.semantic_search import load_search_service
from rag_pipeline.tokenization import EmbeddingTokenizer

QUESTIONS = Path("data/evaluation/questions.jsonl")
CORPUS_LOCK = Path("data/evaluation/corpus_lock.json")


def identity(settings: Settings, corpus: str) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "collection": settings.qdrant_collection,
        "corpus_fingerprint": corpus,
        "processing_fingerprint": processing_fingerprint(settings, embedding_assets_id(settings)),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "generation_model": settings.ollama_model,
        "num_ctx": settings.ollama_num_ctx,
        "temperature": settings.rag_temperature,
    }


def checked_questions(settings: Settings):
    questions = load_questions(QUESTIONS)
    asset = resolve_embedding_tokenizer(settings)
    report = prepare_documents(
        settings, EmbeddingTokenizer(asset.tokenizer_file), embedding_assets_id(settings)
    )
    if report.failures:
        raise EvaluationError("Sample documents failed to prepare; fix them before evaluating")
    documents = [prepared.document for prepared in report.documents]
    validate_labels(questions, documents, read_corpus_lock(CORPUS_LOCK))
    return questions, corpus_fingerprint(documents)


def write_report(output: Path, kind: str, report: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output / f"{kind}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def print_retrieval(report: dict) -> None:
    results = report["results"]
    overall = results["overall"]
    print(f"Retrieval, {overall['questions']} answerable questions, K={results['k']}, ", end="")
    print(f"chunk {report['identity']['chunk_size']}/{report['identity']['chunk_overlap']}")
    print(
        f"  Hit Rate@K {percent(overall['hit_rate'])}, Recall@K {percent(overall['recall'])}, "
        f"MRR@K {overall['mrr']:.3f}, top document {percent(overall['top_1_document'])}"
    )
    for name, values in results["by_category"].items():
        print(
            f"  {name}: hit {percent(values['hit_rate'])}, recall {percent(values['recall'])}, "
            f"MRR {values['mrr']:.3f} (n={values['questions']})"
        )
    latency = results["latency"]
    print(f"  Latency: mean {latency['mean_ms']:.1f} ms, p95 {latency['p95_ms']:.1f} ms")


def print_rag(summary: dict) -> None:
    citations = summary["citations"]
    refusals = summary["refusals"]
    evidence = summary["evidence"]
    print(f"RAG, {summary['questions']} questions. Statuses: {summary['status_counts']}")
    print(
        f"  Citation validity {percent(citations['validity'])} "
        f"({citations['valid_references']}/{citations['emitted_references']}); "
        f"uncited substantive answers {citations['uncited_substantive_answers']}"
    )
    print(
        f"  Correct refusals {refusals['correct_refusals']}/"
        f"{refusals['unanswerable_questions']}; incorrect refusals "
        f"{refusals['incorrect_refusals']}/{refusals['answerable_questions']}"
    )
    print(
        f"  Multi-passage evidence: all retrieved {evidence['all_retrieved']}/"
        f"{evidence['questions']}, all retained {evidence['all_retained']}/{evidence['questions']}"
    )
    print(f"  Correctness: {summary['correctness']}")
    print(f"  Failure stages: {summary['failure_stages'] or 'none'}")
    latency = summary["generation_latency"]
    if latency:
        print(f"  Generation: mean {latency['mean_ms']:.0f} ms, p95 {latency['p95_ms']:.0f} ms")


def run_retrieval(settings: Settings, args: argparse.Namespace) -> int:
    questions, corpus = checked_questions(settings)
    results = evaluate_retrieval(load_search_service(settings), questions, args.top_k)
    report = {"identity": identity(settings, corpus), "results": results}
    path = write_report(args.output, "retrieval", report)
    print_retrieval(report)
    print(f"Report: {path}")
    return 0


def run_rag(settings: Settings, args: argparse.Namespace) -> int:
    questions, corpus = checked_questions(settings)
    rows = evaluate_rag(load_rag_service(settings), questions, args.top_k)
    summary = summarize(rows, questions)
    report = {
        "identity": identity(settings, corpus),
        "top_k": args.top_k,
        "summary": summary,
        "rows": rows_to_json(rows),
    }
    path = write_report(args.output, "rag", report)
    template = {
        "reviewer": "",
        "reviewer_kind": "human",
        "report": path.name,
        "judgments": {row.id: {"score": None, "note": ""} for row in rows},
    }
    template_path = path.with_suffix(".judgments.json")
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print_rag(summary)
    print(f"Report: {path}\nJudgment template: {template_path}")
    return 0


def run_aggregation(args: argparse.Namespace) -> int:
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        data = json.loads(args.judgments.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvaluationError("Report or judgments file is unreadable") from error
    rows = rows_from_json(report["rows"])
    scores = validate_judgments(data, rows)
    summary = summarize(rows, load_questions(QUESTIONS), scores)
    summary["correctness"]["reviewer"] = data["reviewer"]
    summary["correctness"]["reviewer_kind"] = data["reviewer_kind"]
    path = write_report(args.output, "rag-reviewed", {**report, "summary": summary})
    print_rag(summary)
    print(f"Report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("retrieval", "rag"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--report", type=Path, help="existing RAG report to aggregate")
    parser.add_argument("--judgments", type=Path, help="reviewed judgment file")
    args = parser.parse_args()
    if bool(args.report) != bool(args.judgments):
        parser.error("--report and --judgments must be used together")
    if bool(args.mode) == bool(args.report):
        parser.error("choose either --mode or --report with --judgments")

    settings = Settings()
    configure_logging(settings.log_level)
    try:
        if args.report:
            return run_aggregation(args)
        return (
            run_retrieval(settings, args) if args.mode == "retrieval" else run_rag(settings, args)
        )
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
