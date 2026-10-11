"""ChEMBL REST client that caches every response with its URL and retrieval time."""
import hashlib
import json
import time
from pathlib import Path
import requests

API = 'https://www.ebi.ac.uk/chembl/api/data'


class CachedChEMBL:
    def __init__(self, cache=Path('data/raw/chembl')):
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint, **params):
        """Cached GET; reruns are offline once a URL has been fetched."""
        url = f'{API}/{endpoint}.json?' + '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest()[:24] + '.json')
        if path.exists():
            return json.loads(path.read_text())['response']
        for attempt in range(4):
            try:
                reply = requests.get(url, timeout=120); reply.raise_for_status(); break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        path.write_text(json.dumps({'url': url, 'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                    'response': reply.json()}))
        time.sleep(.2)
        return reply.json()

    def pages(self, endpoint, key, max_pages=None, **params):
        """Follow offset pagination; yields records."""
        offset, page = 0, 0
        while True:
            reply = self.get(endpoint, offset=offset, **params)
            yield from reply[key]
            page += 1
            if not reply['page_meta']['next'] or (max_pages and page >= max_pages):
                return
            offset += reply['page_meta']['limit']
