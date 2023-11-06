# coding=utf-8
from logging import Logger
import logging
from logging.handlers import RotatingFileHandler
_logger: Logger = logging.getLogger()


def init_logger(name, debug=False) -> Logger:
    global _logger
    _logger = Logger(name, logging.DEBUG)
    handler = RotatingFileHandler(
        'logs/{}-app.log'.format(name), maxBytes=1000000, backupCount=10)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    if debug:
        # 如果不处于调试模式，将日志输出到 stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        _logger.addHandler(stream_handler)
    return _logger


def get_logger() -> Logger:
    global _logger
    return _logger