"""Wrapper around the Pinecone vector database over VectorDB"""
import time
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
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.table_name = collection_name
        self.case_config = db_case_config
        self.db_config = db_config

        if drop_old:
            self._drop_table()
            self._create_table()

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        conn = self._ensure_connection()
        cursor = conn.cursor()

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

    def _ensure_connection(self) -> (Connection):
        conn = pymysql.connect(**self.db_config)
        conn.autocommit = False
        return conn

    def _drop_table(self):
        conn = self._ensure_connection()
        with conn.cursor() as cursor:
            try:
                cursor.execute(f'DROP TABLE IF EXISTS {self.table_name};')
                conn.commit()
            except Exception as e:
                log.warning(f"Failed to drop pgvector table: {self.table_name} error: {e}")
                raise e from None

    def _create_table(self):
        conn = self._ensure_connection()
        index_param = self.case_config.index_param()
        with conn.cursor() as cursor:
            try:
                cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.table_name} (id BIGINT PRIMARY KEY, embedding vector<float>({self.dim}) COMMENT "hnsw(distance={index_param["metric"]})" );')
                conn.commit()
            except Exception as e:
                log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
                raise e from None

    def ready_to_load(self):
        pass

    def optimize(self):
        while True:
            progress = self._check_tiflash_replica_progress()
            if progress != 1:
                log.info(f"TiFlash still not ready, progress: {progress}")
                time.sleep(2)
            else:
                break
        # log.info("Begin to compact tiflash replica")
        # self._compact_tiflash()
        # log.info("Successful compacted tiflash replica")

    def _compact_tiflash(self):
        conn = self._ensure_connection()
        with conn.cursor() as cursor:
            try:
                cursor.execute(f'ALTER TABLE {self.table_name} COMPACT TIFLASH REPLICA')
                conn.commit()
            except Exception as e:
                log.warning(f"Failed to compact table: {self.table_name} error: {e}")
                raise e from None

    def _check_tiflash_replica_progress(self):
        conn = self._ensure_connection()
        database = self.db_config['database']
        with conn.cursor() as cursor:
            try:
                cursor.execute(f'SELECT PROGRESS FROM information_schema.tiflash_replica WHERE TABLE_SCHEMA = "{database}" AND TABLE_NAME = "{self.table_name}"')
                result = cursor.fetchone()
                return result[0]
            except Exception as e:
                raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        conn = self._ensure_connection()
        with conn.cursor() as cursor:
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
        search_param =self.case_config.search_param()
        conn = self._ensure_connection()
        with conn.cursor() as cursor:
            cursor.execute(f'SELECT id FROM {self.table_name} ORDER BY {search_param["metric_func"]}(embedding, "{query}") LIMIT {k};')
            result = cursor.fetchall()
            return [int(i[0]) for i in result]
