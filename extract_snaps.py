import polars as pl  # noqa: F401
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Extract_snaps():

    def __init__(self) -> None:
        pass
