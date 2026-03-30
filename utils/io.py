import json
import os
from datetime import datetime


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(payload, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def timestamped_run_id(prefix):
    return "%s_%s" % (prefix, datetime.now().strftime("%Y%m%d_%H%M%S"))
