"""Atomic status replacement with bounded retries for Windows reader sharing locks."""
import json
import os
from pathlib import Path
import time


def write_json(path, value, retry_seconds=5):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    deadline = time.monotonic()+retry_seconds
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)
