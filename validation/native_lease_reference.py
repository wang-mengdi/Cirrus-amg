"""Owned CPU sentinel for native process-lease tests; never the real Aphros run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from windows_atomic_json import write_json

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
state = bytearray(bytes(range(256))*262144)
expected = hashlib.sha256(state).hexdigest()
owner = psutil.Process()
record = {'pid': os.getpid(), 'creation_time': owner.create_time(),
          'executable': owner.exe(), 'command': owner.cmdline(),
          'scope': __doc__, 'sentinel_bytes': len(state), 'sentinel_sha256': expected}
(root/'process.json').write_text(json.dumps(record, indent=2)+'\n')
ticks = 0
while not (root/'stop').exists():
    actual = hashlib.sha256(state).hexdigest()
    if actual != expected:
        raise RuntimeError('CPU reference sentinel state changed')
    ticks += 1
    write_json(root/'state.json', {'ticks': ticks, 'sha256': actual, 'bytes': len(state)})
    time.sleep(.2)
(root/'completion.json').write_text(json.dumps({'passed': True, 'ticks': ticks, 'sha256': expected})+'\n')
