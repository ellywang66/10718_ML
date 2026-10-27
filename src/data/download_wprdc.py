"""Download the public WPRDC inspection tables into data/raw/ and record them in data/raw/MANIFEST.json."""
import json
from datetime import date

import requests
from src.config import RAW, WPRDC


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in WPRDC.items():
        path = RAW / name
        if not path.exists():
            print('downloading', name)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        if name not in manifest:
            with open(path, 'rb') as f:
                rows = sum(1 for _ in f) - 1
            downloaded = date.fromtimestamp(path.stat().st_mtime).isoformat()
            manifest[name] = dict(url=url, downloaded=downloaded, bytes=path.stat().st_size, rows=rows)
        print(name, manifest[name])
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
