"""Wrapper around TiDB Serverless"""

import concurrent
import io
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
        self.insert_concurrency = 10

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
        self.query_conn = self._ensure_connection()
        self.query_cursor = self.query_conn.cursor()

        try:
            yield
        finally:
            self.query_cursor.close()
            self.query_conn.close()

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return TiDBServerlessConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig

    def _ensure_connection(self) -> Connection:
        conn = pymysql.connect(**self.db_config)
        conn.autocommit = False
        return conn

    def _drop_table(self):
        with self._ensure_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                    conn.commit()
                except Exception as e:
                    log.warning(f"Failed to drop table: {self.table_name} error: {e}")
                    raise e from None

    def _create_table(self):
        with self._ensure_connection() as conn:
            index_param = self.case_config.index_param()
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        f'CREATE TABLE IF NOT EXISTS {self.table_name} (id BIGINT PRIMARY KEY, embedding VECTOR({self.dim}) COMMENT "hnsw(distance={index_param["metric"]})" );'
                    )
                    conn.commit()
                except Exception as e:
                    log.warning(f"Failed to create table: {self.table_name} error: {e}")
                    raise e from None

    def ready_to_load(self):
        pass

    def optimize(self):
        while True:
            progress = self._check_tiflash_replica_progress()
            if progress != 1:
                log.info(f"Data replication not ready, progress: {progress}")
                time.sleep(2)
            else:
                break

        log.info("Begin compact tiflash replica")
        self._compact_tiflash()
        log.info("Successful compacted tiflash replica")

        while True:
            pending_rows = self._get_tiflash_index_pending_rows()
            if pending_rows > 0:
                log.info(f"Index not fully built, pending rows: {pending_rows}")
                time.sleep(2)
            else:
                break

    def _compact_tiflash(self):
        with self._ensure_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(f"ALTER TABLE {self.table_name} COMPACT")
                    conn.commit()
                except Exception as e:
                    log.warning(
                        f"Failed to compact table: {self.table_name} error: {e}"
                    )
                    raise e from None

    def _check_tiflash_replica_progress(self):
        with self._ensure_connection() as conn:
            database = self.db_config["database"]
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        f'SELECT PROGRESS FROM information_schema.tiflash_replica WHERE TABLE_SCHEMA = "{database}" AND TABLE_NAME = "{self.table_name}"'
                    )
                    result = cursor.fetchone()
                    return result[0]
                except Exception as e:
                    raise e from None

    def _get_tiflash_index_pending_rows(self):
        with self._ensure_connection() as conn:
            database = self.db_config["database"]
            with conn.cursor() as cursor:
                try:
                    cursor.execute(
                        f'SELECT MAX(ROWS_STABLE_NOT_INDEXED) FROM information_schema.tiflash_indexes WHERE TIDB_DATABASE = "{database}" AND TIDB_TABLE = "{self.table_name}"'
                    )
                    result = cursor.fetchone()
                    return result[0]
                except Exception as e:
                    raise e from None

    def _insert_embeddings_serial(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        offset: int,
        size: int,
    ) -> Exception:
        with self._ensure_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    buf = io.StringIO()
                    buf.write(f"INSERT INTO {self.table_name} (id, embedding) VALUES ")
                    for i in range(offset, offset + size):
                        if i > offset:
                            buf.write(",")
                        buf.write(f'({metadata[i]}, "{str(embeddings[i])}")')
                    cursor.execute(buf.getvalue())

                conn.commit()

                return None

            except Exception as e:
                log.warning(
                    f"Failed to insert data into table ({self.table_name}), error: {e}"
                )
                return e

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        insert_batch_size = len(embeddings) // self.insert_concurrency

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.insert_concurrency
        ) as executor:
            futures = []
            for i in range(0, len(embeddings), insert_batch_size):
                offset = i
                size = min(insert_batch_size, len(embeddings) - i)
                future = executor.submit(
                    self._insert_embeddings_serial, embeddings, metadata, offset, size
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                ex = future.result()
                if ex:
                    return 0, ex

        return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        search_param = self.case_config.search_param()

        self.query_cursor.execute(
            f'SELECT id FROM {self.table_name} ORDER BY {search_param["metric_func"]}(embedding, "{query}") LIMIT {k};'
        )
        result = self.query_cursor.fetchall()
        return [int(i[0]) for i in result]
