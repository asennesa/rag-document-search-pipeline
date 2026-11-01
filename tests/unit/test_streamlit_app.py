"""UI behavior with a fake facade. Live scenarios are recorded separately in PLANS.md."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag_pipeline import app_services
from rag_pipeline.app_services import PipelineOverview, UploadResult
from rag_pipeline.errors import DocumentError, VectorStoreError
from rag_pipeline.ingestion.ingestion_pipeline import IngestionReport, PreparationReport
from rag_pipeline.ingestion.synchronization import DocumentResult
from rag_pipeline.models import (
    CitationResult,
    ContextSource,
    RagResponse,
    RetrievedPassage,
    SearchResponse,
)
from rag_pipeline.vector_store.qdrant_store import CollectionStats

APP = str(Path(__file__).parents[2] / "app" / "streamlit_app.py")
ATTACK = "<script>alert(1)</script> **bold** [click](http://evil.test)"


def passage(index=0, text="Employees receive 25 days.", page=None):
    return RetrievedPassage(
        chunk_id=f"chunk-{index}",
        document_id="doc",
        source="annual_leave_policy.md",
        filename="annual_leave_policy.md",
        page_number=page,
        chunk_index=index,
        text=text,
        score=0.8123,
    )


def search_response(results=None):
    return SearchResponse(
        query="Question",
        top_k=5,
        source_filter=None,
        results=(passage(),) if results is None else results,
        embedding_time_ms=1.5,
        vector_query_time_ms=2.5,
        retrieval_time_ms=4.5,
    )


def rag_response(status="answered", answer="You get 25 days.", message=None, citation=True):
    item = passage()
    return RagResponse(
        question="Question",
        status=status,
        answer=answer,
        message=message,
        citation=CitationResult(status="valid", cited_ids=("S1",)) if citation else None,
        sources=(ContextSource(source_id="S1", passage=item),),
        retrieved=(item,),
        model="test-model",
        retrieval_time_ms=5.0,
        generation_time_ms=2000.0,
        total_time_ms=2010.0,
    )


def stats():
    return CollectionStats(
        name="document_embeddings",
        status="green",
        optimizer_status="ok",
        exact_points=24,
        approximate_points=24,
        indexed_vectors=0,
        segments=6,
        dimensions=384,
        distance="Cosine",
        hnsw_m=16,
        hnsw_ef_construct=100,
        full_scan_threshold_kb=10000,
        indexing_threshold_kb=20000,
        payload_indexes=("source",),
    )


def ingestion_report(results=(), failures=()):
    preparation = PreparationReport("fp", (), failures, 1.0)
    failed = tuple(result for result in results if result.status == "failed")
    return IngestionReport(preparation, tuple(results), (), (), failed_results=failed)


class FakeFacade:
    def __init__(self):
        self.search_error = None
        self.ask_error = None
        self.overview_error = None
        self.sources_error = None
        self.search_result = search_response()
        self.rag_result = rag_response()
        self.ingest_result = ingestion_report([DocumentResult("a.md", "synced", 2, 0)])
        self.exists = True
        self.calls = []

    def search(self, question, top_k, source):
        self.calls.append(("search", question, top_k, source))
        if self.search_error:
            raise self.search_error
        return self.search_result

    def ask(self, question, top_k, source):
        self.calls.append(("ask", question, top_k, source))
        if self.ask_error:
            raise self.ask_error
        return self.rag_result

    def sources(self):
        if self.sources_error:
            raise self.sources_error
        return ["annual_leave_policy.md", "remote_work_policy.md"]

    def overview(self):
        if self.overview_error:
            raise self.overview_error
        return PipelineOverview(
            collection_exists=self.exists,
            stats=stats() if self.exists else None,
            document_count=12,
            uploaded_document_count=1,
            embedding_model="BAAI/bge-small-en-v1.5",
            interpretation="HNSW is configured, not in use." if self.exists else None,
            hnsw_ef=128,
        )

    def ingest_samples(self):
        self.calls.append(("ingest",))
        return self.ingest_result


@pytest.fixture
def facade(monkeypatch):
    fake = FakeFacade()
    st.cache_resource.clear()
    monkeypatch.setattr(app_services, "create_facade", lambda _settings: fake)
    yield fake
    st.cache_resource.clear()


def run_app():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    return app


def submit(app, key, question, *, top_k=None, source=None):
    app.text_input(key=f"{key}_question").set_value(question)
    if top_k is not None:
        app.slider(key=f"{key}_top_k").set_value(top_k)
    if source is not None:
        app.selectbox(key=f"{key}_source").set_value(source)
    index = 0 if key == "search" else 1
    return app.button[index].click().run()


def all_text(app):
    return "\n".join(
        element.value for group in (app.text, app.code, app.caption) for element in group
    )


def test_all_three_sections_render(facade):
    app = run_app()
    assert [tab.label for tab in app.tabs] == [
        "Semantic Search",
        "RAG Assistant",
        "Pipeline Inspection",
    ]


def test_loading_the_page_runs_no_query_or_ingestion(facade):
    run_app()
    assert facade.calls == []


def test_search_passes_controls_and_shows_results(facade):
    app = submit(run_app(), "search", "How long?", top_k=3, source="annual_leave_policy.md")
    assert facade.calls == [("search", "How long?", 3, "annual_leave_policy.md")]
    text = all_text(app)
    assert "Employees receive 25 days." in text
    assert "annual_leave_policy.md, chunk 0" in text
    assert "Score: 0.8123" in text
    assert "Retrieval 4.5 ms" in text
    assert "not confidence" in text


def test_all_documents_means_no_filter(facade):
    submit(run_app(), "search", "Question")
    assert facade.calls[0][3] is None


def test_top_k_selector_uses_configured_range(facade):
    slider = run_app().slider(key="search_top_k")
    assert (slider.min, slider.max, slider.value) == (1, 20, 5)


def test_page_numbers_appear_for_pdf_passages(facade):
    facade.search_result = search_response((passage(page=3),))
    assert ", page 3, chunk 0" in all_text(submit(run_app(), "search", "Question"))


def test_empty_results_are_explained(facade):
    facade.search_result = search_response(())
    app = submit(run_app(), "search", "Question")
    assert "No matching passages." in [item.value for item in app.info]


def test_validation_and_store_errors_are_shown(facade):
    facade.search_error = VectorStoreError("Qdrant is unreachable; is it running?")
    app = submit(run_app(), "search", "Question")
    assert [item.value for item in app.error] == ["Qdrant is unreachable; is it running?"]
    assert not app.exception


def test_failed_search_clears_the_previous_result(facade):
    app = submit(run_app(), "search", "First")
    facade.search_error = VectorStoreError("down")
    app = submit(app, "search", "Second")
    assert "Employees receive 25 days." not in all_text(app)


def test_answer_and_evidence_are_separate(facade):
    app = submit(run_app(), "rag", "How long?")
    assert [item.value for item in app.subheader][:2] == ["Generated answer", "Retrieved passages"]
    assert "You get 25 days." in [item.value for item in app.text]
    assert "[S1] annual_leave_policy.md" in [item.value for item in app.text]
    assert "not generated by the model" in all_text(app)
    assert [item.label for item in app.expander] == ["Passage 1 (score 0.8123)"]
    assert "LLM generation 2000.0 ms" in all_text(app)
    assert "not that it is true" in app.success[0].value


def test_invalid_citations_are_flagged_and_list_no_sources(facade):
    facade.rag_result = rag_response(
        "citation_invalid", "Unsupported claim.", "Do not treat it as grounded"
    )
    app = submit(run_app(), "rag", "Question")
    assert "Do not treat it as grounded" in app.warning[0].value
    assert not app.success
    assert "[S1] annual_leave_policy.md" not in [item.value for item in app.text]


def test_refusal_is_not_shown_as_success(facade):
    facade.rag_result = rag_response(
        "insufficient_evidence", "The documents do not cover it.", citation=False
    )
    app = submit(run_app(), "rag", "Question")
    assert not app.success
    assert "do not answer this" in app.info[0].value


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("ollama_unavailable", "Ollama is unreachable; is it running? Try: ollama serve"),
        ("model_missing", "Model is not installed. Run: ollama pull x"),
        ("timeout", "Ollama did not answer in time"),
    ],
)
def test_generation_failures_explain_and_keep_passages(facade, status, message):
    facade.rag_result = rag_response(status, None, message, citation=False)
    app = submit(run_app(), "rag", "Question")
    assert message in app.warning[0].value
    assert "Employees receive 25 days." in all_text(app)


def test_search_still_works_when_rag_fails(facade):
    facade.ask_error = VectorStoreError("rag down")
    app = submit(run_app(), "rag", "Question")
    assert app.error[0].value == "rag down"
    app = submit(app, "search", "Question")
    assert "Employees receive 25 days." in all_text(app)


def test_untrusted_text_never_reaches_markdown(facade):
    item = passage(text=ATTACK)
    facade.search_result = search_response((item,))
    facade.rag_result = rag_response(answer=ATTACK)
    app = submit(run_app(), "search", "Question")
    app = submit(app, "rag", "Question")
    rendered_as_markdown = [element.value for element in app.markdown]
    assert not any("script" in value or "evil.test" in value for value in rendered_as_markdown)
    assert ATTACK in [element.value for element in app.code]
    assert ATTACK in [element.value for element in app.text]


def test_inspection_shows_observed_state(facade):
    app = run_app()
    assert [(m.label, m.value) for m in app.metric] == [
        ("Documents", "12"),
        ("Uploaded documents", "1"),
        ("Chunks", "24"),
    ]
    table = app.table[0].value
    values = dict(zip(table["Setting"], table["Value"], strict=True))
    assert values["Embedding model"] == "BAAI/bge-small-en-v1.5"
    assert values["Vector dimensions"] == "384"
    assert values["Distance metric"] == "Cosine"
    assert values["Indexed vectors"] == "0"
    assert values["HNSW m"] == "16"
    assert "not in use" in app.info[0].value


def test_empty_collection_guides_the_user(facade):
    facade.exists = False
    app = run_app()
    assert "does not exist yet" in app.info[0].value
    assert not app.metric


def test_outage_in_inspection_is_reported_and_other_tabs_load(facade):
    facade.overview_error = VectorStoreError("Qdrant is unreachable")
    facade.sources_error = VectorStoreError("Qdrant is unreachable")
    app = run_app()
    assert [item.value for item in app.error] == ["Qdrant is unreachable"]
    assert app.selectbox(key="search_source").options == ["All documents"]


def test_sample_ingestion_reports_success_only_after_it_finishes(facade):
    app = run_app()
    assert not app.success
    app = app.button[2].click().run()
    assert ("ingest",) in facade.calls
    assert app.success[0].value == "Sample ingestion finished."
    assert "Synced 1, unchanged 0" in all_text(app)


def test_partial_ingestion_is_not_called_success(facade):
    facade.ingest_result = ingestion_report(
        [DocumentResult("a.md", "failed", error_category="store_failed", message="write failed")]
    )
    app = run_app().button[2].click().run()
    assert not app.success
    assert app.warning[0].value == "Ingestion finished with failures."
    assert "Failed: a.md: write failed" in all_text(app)


def test_ingestion_error_is_shown(facade):
    def fail():
        raise VectorStoreError("Another ingestion is running")

    facade.ingest_samples = fail
    app = run_app().button[2].click().run()
    assert app.error[0].value == "Another ingestion is running"


def upload_file(app, name, data):
    app.file_uploader[0].set_value((name, data, "text/markdown"))
    return app.run()


def click_upload(app):
    return next(b for b in app.button if b.label == "Ingest uploaded file").click().run()


def test_upload_button_appears_only_with_a_file(facade):
    app = run_app()
    assert "Ingest uploaded file" not in [button.label for button in app.button]
    app = upload_file(app, "note.md", b"text")
    assert "Ingest uploaded file" in [button.label for button in app.button]


def test_upload_reports_success_after_it_finishes(facade):
    def ingest_upload(name, data):
        facade.calls.append(("upload", name, data))
        return UploadResult("abc-note.md", "synced", 3, facade.ingest_result)

    facade.ingest_upload = ingest_upload
    app = click_upload(upload_file(run_app(), "note.md", b"text"))
    assert ("upload", "note.md", b"text") in facade.calls
    assert app.success[0].value == "Upload ingested."
    assert "abc-note.md: synced, 3 chunks" in all_text(app)


def test_rejected_upload_shows_the_reason(facade):
    def ingest_upload(name, data):
        raise DocumentError("corrupt_pdf", "PDF cannot be parsed")

    facade.ingest_upload = ingest_upload
    app = click_upload(upload_file(run_app(), "note.md", b"text"))
    assert app.error[0].value == "PDF cannot be parsed"
    assert not app.success


def test_failed_upload_is_not_called_success(facade):
    facade.ingest_upload = lambda name, data: UploadResult(
        "abc.md", "failed", 0, facade.ingest_result
    )
    app = click_upload(upload_file(run_app(), "note.md", b"text"))
    assert app.warning[0].value == "The upload was not ingested."
    assert not app.success


def test_invalid_configuration_is_reported_without_details(facade, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    app = run_app()
    assert "Configuration is invalid" in app.error[0].value
    assert "not-a-number" not in app.error[0].value
