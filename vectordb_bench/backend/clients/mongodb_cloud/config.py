from pydantic import SecretStr, BaseModel
from ..api import DBConfig, DBCaseConfig, IndexType, MetricType


class MongodbCloudConfig(DBConfig):
    connection_string: SecretStr
    database: str

    def to_dict(self) -> dict:
        return {
            "connection_string": self.connection_string.get_secret_value(),
            "database": self.database,
        }


class MongodbCloudIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        elif self.metric_type == MetricType.IP:
            return "dotProduct"
        return "cosine"

    def index_param(self) -> dict:
        params = {
            "similarity": self.parse_metric(),
        }
        return params

    def search_param(self) -> dict:
        return {}
