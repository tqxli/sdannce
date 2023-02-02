import logging
import logging.config
from pathlib import Path
import json
from collections import OrderedDict
import os


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

log_config = OrderedDict({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(message)s"
        },
        "datetime": {
            "format": "[%(asctime)s][%(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "datetime",
            "filename": "training.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "info_file_handler"]
    }
})


def setup_logging(basedir, filename):
    """
    Setup logging configuration
    """
    for _, handler in log_config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = os.path.join(basedir, filename)
    logging.config.dictConfig(log_config)

    logger = logging.getLogger(__name__)
    return logger
