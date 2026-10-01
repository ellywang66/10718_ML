"""Download the public WPRDC inspection tables into data/raw/."""
import requests
from config import RAW, WPRDC


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in WPRDC.items():
        path = RAW / name
        if path.exists():
            print('exists', path); continue
        print('downloading', name)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print('saved', path, path.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
