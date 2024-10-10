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
