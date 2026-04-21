"""Tests for Milvus ingestion helpers."""

from typing import TYPE_CHECKING, cast

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.rag.chunking import KnowledgeChunk
from enterprise_knowledge_assistant.rag.ingestion import (
    ChunkEmbeddingRecord,
    build_milvus_record,
    ingest_documents,
)

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.rag.ingestion import (
        MilvusIngestionClientProtocol,
    )


class FakeSchema:
    """Fake schema object used to capture Milvus schema fields in tests."""

    def __init__(self) -> None:
        """Initialize the fake schema field store."""
        self.fields: list[dict[str, object]] = []

    def add_field(self, **field_kwargs: object) -> None:
        """Record a schema field definition."""
        self.fields.append(field_kwargs)


class FakeIndexParams:
    """Fake index params object used to capture Milvus index config."""

    def __init__(self) -> None:
        """Initialize the fake index parameter store."""
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **index_kwargs: object) -> None:
        """Record an index definition."""
        self.indexes.append(index_kwargs)


class FakeMilvusClient:
    """Simple fake to capture collection creation and upsert calls."""

    def __init__(self) -> None:
        """Initialize the fake Milvus client state."""
        self.upsert_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.created_collection_names: list[str] = []
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()

    def has_collection(self, *, collection_name: str) -> bool:
        """Pretend the collection does not yet exist."""
        return False

    def create_schema(
        self,
        *,
        auto_id: bool,
        enable_dynamic_fields: bool,
    ) -> FakeSchema:
        """Return a fake schema while recording its creation config."""
        self.schema_config = {
            "auto_id": auto_id,
            "enable_dynamic_fields": enable_dynamic_fields,
        }
        return self.schema

    def prepare_index_params(self) -> FakeIndexParams:
        """Return the fake index parameter collector."""
        return self.index_params

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: object,
        index_params: object,
    ) -> None:
        """Record collection creation inputs."""
        self.created_collection_names.append(collection_name)
        self.created_schema = schema
        self.created_index_params = index_params

    def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        """Record an upsert call."""
        self.upsert_calls.append((collection_name, data))


def test_build_milvus_record_maps_chunk_fields() -> None:
    """A chunk embedding record should be converted to a Milvus entity payload."""
    chunk = KnowledgeChunk(
        chunk_id="hr/remote_work_policy.md::chunk-000",
        document="remote_work_policy.md",
        category="hr",
        path="hr/remote_work_policy.md",
        title="Remote Work Policy",
        text="Employees may work remotely up to three days.",
    )
    record = ChunkEmbeddingRecord(chunk=chunk, embedding=[0.1, 0.2, 0.3])

    payload = build_milvus_record(record)

    assert payload == {
        "id": "hr/remote_work_policy.md::chunk-000",
        "vector": [0.1, 0.2, 0.3],
        "document": "remote_work_policy.md",
        "category": "hr",
        "path": "hr/remote_work_policy.md",
        "title": "Remote Work Policy",
        "text": "Employees may work remotely up to three days.",
    }


def test_ingest_documents_creates_collection_and_upserts_records() -> None:
    """Ingestion should ensure the collection exists before upserting data."""
    chunk = KnowledgeChunk(
        chunk_id="it/vpn_access_guide.md::chunk-000",
        document="vpn_access_guide.md",
        category="it",
        path="it/vpn_access_guide.md",
        title="VPN Access Guide",
        text="VPN access requires MFA and an approved device.",
    )
    record = ChunkEmbeddingRecord(chunk=chunk, embedding=[0.1, 0.2, 0.3, 0.4])
    client = FakeMilvusClient()
    settings = Settings(
        milvus_collection_name="knowledge_chunks",
        milvus_embedding_dimension=4,
    )

    result = ingest_documents(
        [record],
        settings=settings,
        client=cast("MilvusIngestionClientProtocol", client),
    )

    assert result == {"inserted_count": 1}
    assert client.created_collection_names == ["knowledge_chunks"]
    assert client.upsert_calls == [
        (
            "knowledge_chunks",
            [
                {
                    "id": "it/vpn_access_guide.md::chunk-000",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "document": "vpn_access_guide.md",
                    "category": "it",
                    "path": "it/vpn_access_guide.md",
                    "title": "VPN Access Guide",
                    "text": "VPN access requires MFA and an approved device.",
                },
            ],
        ),
    ]
    assert client.schema_config == {
        "auto_id": False,
        "enable_dynamic_fields": False,
    }
    assert len(client.schema.fields) == 7
    assert client.index_params.indexes == [
        {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        },
    ]


def test_ingest_documents_returns_zero_for_empty_payload() -> None:
    """Ingestion should no-op cleanly when no records are provided."""

    class FakeMilvusClient:
        """Simple fake used to verify no upsert happens."""

        def has_collection(self, *, collection_name: str) -> bool:
            """Pretend the collection already exists."""
            return True

        def upsert(
            self,
            *,
            collection_name: str,
            data: list[dict[str, object]],
        ) -> None:
            """Fail if upsert is attempted for an empty payload."""
            msg = "upsert should not be called for an empty payload"
            raise AssertionError(msg)

    result = ingest_documents(
        [],
        client=cast("MilvusIngestionClientProtocol", FakeMilvusClient()),
    )

    assert result == {"inserted_count": 0}
