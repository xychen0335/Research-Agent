from research_agent.data.papersearchqa import resolve_parquet, resolve_pubmed_jsonl
from research_agent.paths import DEFAULT_MODEL, model_available, resolve_model_path

from tests.support import TempDirTestCase


class TestWorkspacePaths(TempDirTestCase):
    def test_resolve_model_path_prefers_existing_directory(self):
        dest = self.tmp_path / "Qwen3.5-4B"
        dest.mkdir()
        (dest / "config.json").write_text("{}", encoding="utf-8")
        assert resolve_model_path(str(dest)) == dest
        assert model_available(str(dest)) is True

    def test_missing_model_is_not_available(self):
        assert model_available(str(self.tmp_path / "missing-model")) is False
        assert DEFAULT_MODEL.name == "Qwen3.5-4B"

    def test_resolve_parquet_uses_local_file_only(self):
        raw = self.tmp_path / "papersearchqa"
        raw.mkdir()
        parquet = raw / "train-00000-of-00001.parquet"
        parquet.write_bytes(b"not-a-real-parquet")
        assert resolve_parquet("train", raw) == parquet
        try:
            resolve_parquet("test", raw)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert "test" in str(exc)

    def test_resolve_pubmed_jsonl_uses_local_file_only(self):
        raw = self.tmp_path / "pubmed_bioasq_2022"
        raw.mkdir()
        dump = raw / "pubmed.jsonl"
        dump.write_text("{}\n", encoding="utf-8")
        assert resolve_pubmed_jsonl(raw) == dump
        try:
            resolve_pubmed_jsonl(self.tmp_path / "empty")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert "pubmed.jsonl" in str(exc)

    def test_fetch_abstracts_reads_local_cache_only(self):
        from research_agent.data.pubmed import fetch_abstracts

        cache = self.tmp_path / "abstracts.json"
        cache.write_text(
            '{"1": {"title": "t", "abstract": "RB1"}, "2": {"title": "x"}}\n',
            encoding="utf-8",
        )
        found = fetch_abstracts(["1", "2", "3"], cache_path=cache)
        assert list(found) == ["1"]
        assert found["1"]["abstract"] == "RB1"
