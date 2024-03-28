"""Wrapper around the Pinecone vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import Type
import pymysql
from pymysql.cursors import Cursor
from pymysql.connections import Connection

from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import TiDBServerlessConfig


log = logging.getLogger(__name__)


class TiDBServeless(VectorDB):
    name = "TiDBServerless"

    def __init__(
        self,
        dim,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "tidb_serverless_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.table_name = collection_name
        self.case_config = db_case_config
        self.db_config = db_config

        # if drop_old:
        #     self._drop_table()
        #     self._create_table()

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        conn, cursor = self._ensure_connection()

        try:
            yield
        finally:
            cursor.close()
            conn.close()

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return TiDBServerlessConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig

    def _ensure_connection(self) -> (Connection, Cursor):
        conn = pymysql.connect(**self.db_config)
        conn.autocommit = False
        cursor = conn.cursor()
        return conn, cursor

    def _drop_table(self):
        conn, cursor = self._ensure_connection()
        try:
            cursor.execute(f'DROP TABLE IF EXISTS {self.table_name};')
            conn.commit()
        except Exception as e:
            log.warning(f"Failed to drop pgvector table: {self.table_name} error: {e}")
            raise e from None

    def _create_table(self):
        conn, cursor = self._ensure_connection()
        index_param = self.case_config.index_param()
        try:
            # create table
            cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.table_name} (id BIGINT PRIMARY KEY, embedding vector<float>({self.dim}) COMMENT "hnsw(distance={index_param["metric"]})" );')
            conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

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
        return len(metadata), None
        conn, cursor = self._ensure_connection()
        try:
            batch_size = 5000
            for i in range(0, len(metadata), batch_size):
                batch_ids = metadata[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                if len(batch_ids) == 0:
                    break

                batch_embeddings = list(map(lambda x: str(list(x)), batch_embeddings))

                cursor.executemany(f'INSERT INTO {self.table_name} (id, embedding) VALUES (%s, %s)', list(zip(batch_ids, batch_embeddings)))
                conn.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")   
            return 0, e

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        conn, cursor = self._ensure_connection()
        search_param =self.case_config.search_param()
        cursor.execute(f'SELECT id FROM {self.table_name} ORDER BY {search_param["metric_func"]}(embedding, "{query}") LIMIT {k};')
        result = cursor.fetchall()
        return [int(i[0]) for i in result]
