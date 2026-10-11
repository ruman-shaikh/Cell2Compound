"""Verify the scPerturb checksum before making the download usable."""
import hashlib
import json
from pathlib import Path

metadata = json.loads(Path('data/raw/scperturb_record.json').read_text())
record = next(f for f in metadata['files'] if f['key'] == 'SrivatsanTrapnell2020_sciplex3.h5ad')
source = Path('data/raw/sciplex3.h5ad.part')
digest = hashlib.md5()
with source.open('rb') as handle:
    for block in iter(lambda: handle.read(8 * 1024**2), b''):
        digest.update(block)
actual = 'md5:' + digest.hexdigest()
if source.stat().st_size != record['size'] or actual != record['checksum']:
    raise ValueError(f'Download failed integrity verification: {actual}')
source.rename('data/raw/sciplex3.h5ad')
Path('data/sciplex_source.json').write_text(json.dumps({
    'record': 'https://zenodo.org/records/13350497',
    'file': record['key'], 'bytes': record['size'], 'checksum': actual,
    'release_note': 'v1.4 corrects sci-Plex3 errors',
}, indent=2) + '\n')
print('Verified download size and checksum')
