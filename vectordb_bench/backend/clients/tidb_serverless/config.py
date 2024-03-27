from pydantic import SecretStr, BaseModel
from ..api import DBConfig, DBCaseConfig, IndexType, MetricType

class TiDBServerlessConfig(DBConfig):
    host: str = "gateway01.us-east-1.dev.shared.aws.tidbcloud.com"
    user_name: SecretStr = "3SHHfyhwA3JfDph.root"
    password: SecretStr
    port: int = 4000
    db_name: str = "test"

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "user": self.user_name.get_secret_value(),
            "password": self.password.get_secret_value(),
            "port": self.port,
            "database": self.db_name,
            "ssl_verify_cert": True,
            "ssl_verify_identity": True,
        }


class TiDBServerlessIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        return "cosine"

    def parse_metric_fun_str(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vec_l2_distance"
        return "vec_cosine_distance"

    def index_param(self) -> dict:
        return {
            "metric" : self.parse_metric()
        }
    
    def search_param(self) -> dict:
        return {
            "metric_func" : self.parse_metric_fun_str()
        }
