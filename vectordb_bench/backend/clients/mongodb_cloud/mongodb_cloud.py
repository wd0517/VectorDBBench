"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Type
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import MongodbCloudConfig


log = logging.getLogger(__name__)


class MongodbCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MongodbCloudIndexConfig,
        collection_name: str = "test",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.dim = dim
        self.case_config = db_case_config
        self.database = db_config["database"]
        self.collection_name = collection_name
        self.connection_string = db_config["connection_string"]
        self.case_config = db_case_config

        client = MongoClient(self.connection_string, server_api=ServerApi("1"))
        client.admin.command("ping")
        db = client.get_database(self.database)
        if drop_old:
            db.drop_collection(collection_name)

            db.create_collection(collection_name)
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": self.dim,
                            "path": "embedding",
                            "similarity": self.case_config.index_param()["similarity"],
                        }
                    ]
                },
                name="vector_index",
                type="vectorSearch",
            )
            db.get_collection(collection_name).create_search_index(
                model=search_index_model
            )

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return MongodbCloudConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self) -> None:
        self.client = MongoClient(self.connection_string, server_api=ServerApi("1"))
        self.collection = self.client.get_database(self.database).get_collection(
            self.collection_name
        )
        yield

    def ready_to_load(self):
        pass

    def optimize(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        assert len(embeddings) == len(metadata)
        insert_count = 0
        batch_size = 1000
        try:
            for batch_start_offset in range(0, len(embeddings), 1000):
                batch_end_offset = min(batch_start_offset + batch_size, len(embeddings))
                insert_datas = []
                for i in range(batch_start_offset, batch_end_offset):
                    insert_data = {
                        "_id": metadata[i],
                        "embedding": embeddings[i],
                    }
                    insert_datas.append(insert_data)
                self.collection.insert_many(insert_datas)
                insert_count += batch_end_offset - batch_start_offset
        except Exception as e:
            return (insert_count, e)
        return (len(embeddings), None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[tuple[int, float]]:
        search_param = self.case_config.search_param()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query,
                    "numCandidates": k,
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                }
            },
        ]
        result = self.collection.aggregate(pipeline)
        return [int(r["_id"]) for r in result]
