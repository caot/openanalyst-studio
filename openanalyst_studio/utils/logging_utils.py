from __future__ import annotations

import logging
import sys
import time
import uuid

import structlog


def configure_logging(level: int=logging.INFO, json_logs: bool=True) -> None:
    """Call once at app startup."""
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)

    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),  # renders stack when stack_info=True
        structlog.processors.format_exc_info,  # <-- needs exc_info in the event
        structlog.processors.JSONRenderer(sort_keys=True) if json_logs
        else structlog.dev.ConsoleRenderer()
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),  # prints to stdout
        cache_logger_on_first_use=True,
    )

def tool_logger(tool: str):
    return structlog.get_logger().bind(tool=tool)

class timed:
    """Context manager that logs tool start/end + full traceback on error."""
    def __init__(self, log, **fields):
        self.log = log.bind(**fields)
    def __enter__(self):
        self.start = time.perf_counter()
        self.run_id = uuid.uuid4().hex
        self.log.info("tool_start", run_id=self.run_id)
        return self
    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self.start
        if exc is not None:
            # IMPORTANT: use .exception to attach exc_info so format_exc_info can render it
            self.log.exception("tool_error", run_id=self.run_id, duration_s=dur)
            # return False to re-raise if you want upstream handlers; True to swallow.
            return False
        self.log.info("tool_end", run_id=self.run_id, duration_s=dur)
        return False
