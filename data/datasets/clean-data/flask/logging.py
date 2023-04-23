import logging
import sys
from werkzeug.local import LocalProxy
from .globals import request
@LocalProxy
def wsgi_errors_stream():
    return request.environ["wsgi.errors"] if request else sys.stderr
def has_level_handler(logger):
    level = logger.getEffectiveLevel()
    current = logger
    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True
        if not current.propagate:
            break
        current = current.parent
    return False
default_handler = logging.StreamHandler(wsgi_errors_stream)
default_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
)
def create_logger(app):
    logger = logging.getLogger(app.name)
    if app.debug and not logger.level:
        logger.setLevel(logging.DEBUG)
    if not has_level_handler(logger):
        logger.addHandler(default_handler)
    return logger