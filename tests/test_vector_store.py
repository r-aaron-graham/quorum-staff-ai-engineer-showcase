import pytest
from src.vector_store import VectorStore
from sentence_transformers import SentenceTransformer


class DummyModel(SentenceTransformer):
    def __init__(self):
        # No super init
        pass

    def encode(self, texts, show_progress_bar=False):
        # Return fixed embeddings: list of lists
        return [[float(len(t))] for t in texts]

    def get_sentence_embedding_dimension(self):
        return 1


class DummyQdrantClient:
    def __init__(self):
        self.upserted = []
        self.searched = []

    def recreate_collection(self, collection_name, vectors_config):
        # simulate collection creation
        self.collection = collection_name

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def search(self, collection_name, query_vector, limit):
        # simulate returning dummy hits
        class Hit:
            def __init__(self, id, score, payload):
                self.id = id
                self.score = score
                self.payload = payload

        return [Hit(id=1, score=0.9, payload={'text': 'dummy'})]


class DummyOpenSearch:
    def __init__(self):
        self.actions = []
        self.index = {}

    class DummyIndices:
        def __init__(self, parent):
            self.parent = parent

        def exists(self, index_name):
            return False

        def create(self, index, body):
            self.parent.index_name = index

    def search(self, index, body):
        # simulate search response
        return {'hits': {'hits': [{'_id': 2, '_score': 1.0, '_source': {'text': 'foo', 'embedding': [0.1]}}]}}

    @property
    def indices(self):
        return DummyOpenSearch.DummyIndices(self)


def test_qdrant_upsert_and_search(monkeypatch):
    # Patch model and client
    monkeypatch.setattr(VectorStore, 'model', DummyModel())
    monkeypatch.setenv('QDRANT_URL', 'http://fake')
    # Create VectorStore with dummy client
    vs = VectorStore(backend='qdrant', index_name='test')
    vs.client = DummyQdrantClient()

    # Test upsert
    vs.upsert(ids=[1], texts=['hello'])
    assert vs.client.upserted[0][0] == 'test'
    assert vs.client.upserted[0][1][0].payload['text'] == 'hello'

    # Test search
    results = vs.search(query_text='hi', top_k=1)
    assert results[0]['id'] == 1
    assert results[0]['text'] == 'dummy'


def test_opensearch_upsert_and_search(monkeypatch):
    # Patch model
    monkeypatch.setattr(VectorStore, 'model', DummyModel())
    # Create VectorStore with fake OpenSearch client
    vs = VectorStore(backend='opensearch', index_name='test_os')
    vs.client = DummyOpenSearch()

    # Test upsert
    vs.upsert(ids=[2], texts=['world'])
    # verify no exception

    # Test search
    results = vs.search(query_text='anything', top_k=1)
    assert results[0]['id'] == '2'
    assert 'foo' in results[0]['text']
