"""Put an aggregate review archive beside its notebook and display the download."""
import os
from pathlib import Path
import shutil
import tempfile

from IPython.display import FileLink, display


def show_review_zip(archive):
    source = Path(archive).resolve()
    target = Path.cwd() / source.name
    if source != target:
        handle, tmp = tempfile.mkstemp(prefix="." + source.stem, dir=target.parent)
        os.close(handle)
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
    print("Download the return-to-chat ZIP:", target)
    display(FileLink(target.name))
    return target
