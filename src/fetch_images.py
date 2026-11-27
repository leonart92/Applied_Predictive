"""Download and validate the CrisisMMD images referenced by the cleaned splits.

Run from the repository root: python src/fetch_images.py

The official QCRI host stopped answering in November 2026, so the archive is read
from the Internet Archive snapshot of 18 December 2024. Nothing is extracted
before the assembled size matches that snapshot exactly, and every image listed
in splits/ must then open and fully decode. Image validation reuses
prepare_data.check_image, so preparation and training agree on what a usable
image is. Only results/tables/image_fetch_report.json is versioned; data/ is not.
"""
import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import PIL

if __package__:
    from .prepare_data import ROOT, check_image, require
else:
    from prepare_data import ROOT, check_image, require

SNAPSHOT = '20241218122122'
ARCHIVE_URL = ('https://web.archive.org/web/' + SNAPSHOT + 'if_/'
               'https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz')
# Size served by the snapshot, read from a range request before writing this
# script. A short or truncated download must never reach tarfile.
ARCHIVE_BYTES = 1902053684
ARCHIVE_PATH = ROOT / 'data/CrisisMMD_v2.0.tar.gz'
IMAGE_PREFIX = 'CrisisMMD_v2.0/data_image/'
ATTEMPTS = 8
CHUNK = 1 << 20
CONNECTIONS = 8


def split_images():
    """Unique image paths referenced by the six cleaned splits, in sorted order."""
    paths = set()
    for csv_path in sorted((ROOT / 'splits').glob('*.csv')):
        with csv_path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            require('image' in (reader.fieldnames or []), f'No image column: {csv_path}')
            for row in reader:
                require(row['image'].strip(), f'Empty image path: {csv_path}')
                paths.add(row['image'])
    require(paths, 'No image referenced by splits/')
    return sorted(paths)


def fetch_segment(url, destination, start, end, attempts=ATTEMPTS):
    """Download bytes [start, end] into their own part file, resuming it if present."""
    size = end - start + 1
    for attempt in range(1, attempts + 1):
        have = destination.stat().st_size if destination.is_file() else 0
        require(have <= size, f'Part longer than its segment: {destination}')
        if have == size:
            return size
        request = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}'})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                require(response.status == 206, f'Range request refused: HTTP {response.status}')
                with destination.open('ab') as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f'  {destination.name} attempt {attempt}/{attempts}: '
                  f'{type(exc).__name__}: {exc}', flush=True)
    have = destination.stat().st_size if destination.is_file() else 0
    require(have == size, f'Part incomplete after {attempts} attempts: {destination} ({have}/{size})')
    return size


def download(destination=ARCHIVE_PATH, url=ARCHIVE_URL, expected_bytes=ARCHIVE_BYTES,
             connections=CONNECTIONS):
    """Fetch the archive with parallel range requests; return 'cached' or 'downloaded'.

    The snapshot serves a plain GET of this archive at a few hundred kB/s but
    answers concurrent range requests, so the file is cut into contiguous
    segments, each resumable on its own, and assembled once every part is
    complete. Parts are removed only after the assembled size matches.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return 'cached'
    require(isinstance(connections, int) and connections >= 1, 'connections must be a positive integer')
    parts = destination.parent / 'parts'
    parts.mkdir(exist_ok=True)
    span = -(-expected_bytes // connections)
    bounds = [(i * span, min((i + 1) * span, expected_bytes) - 1)
              for i in range(connections) if i * span < expected_bytes]
    paths = [parts / f'{destination.name}.part{i:02d}' for i in range(len(bounds))]
    # An interrupted single-stream download is a prefix of the file, so it can be
    # reused as the beginning of the first segment instead of being thrown away.
    if destination.is_file() and not paths[0].is_file():
        if 0 < destination.stat().st_size <= bounds[0][1] + 1:
            destination.rename(paths[0])
        else:
            destination.unlink()
    print(f'{len(bounds)} segments of about {span} bytes', flush=True)
    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        futures = [pool.submit(fetch_segment, url, path, start, end)
                   for path, (start, end) in zip(paths, bounds)]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f'  {done}/{len(futures)} segments complete', flush=True)
    total = sum(path.stat().st_size for path in paths)
    require(total == expected_bytes, f'Segments total {total} of {expected_bytes} bytes')
    with destination.open('wb') as handle:
        for path in paths:
            with path.open('rb') as part:
                shutil.copyfileobj(part, handle, CHUNK)
    require(destination.stat().st_size == expected_bytes, f'Assembled size mismatch: {destination}')
    for path in paths:
        path.unlink()
    parts.rmdir()
    return 'downloaded'


def extract(archive=ARCHIVE_PATH, target=None, prefix=IMAGE_PREFIX):
    """Stream the archive and write only its data_image subtree under data/.

    Member names are checked before extraction: no absolute path, no parent
    reference, no link or device. Tarfile's data filter is applied as well.
    """
    target = target if target is not None else ROOT / 'data'
    require(archive.stat().st_size == ARCHIVE_BYTES, f'Unexpected archive size: {archive}')
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            path = Path(member.name)
            require(not path.is_absolute() and '..' not in path.parts,
                    f'Unexpected member path: {member.name}')
            require(member.isfile() or member.isdir(), f'Unexpected member type: {member.name}')
            tar.extract(member, path=target, filter='data')
            written += member.isfile()
            if written and written % 4000 == 0:
                print(f'  {written} images extracted', flush=True)
    require(written, f'No member under {prefix} in {archive}')
    return written


def validate(images):
    """Open and fully decode every image the splits reference; report any failure."""
    failures = []
    for i, image in enumerate(images, start=1):
        status, detail = check_image(ROOT / image)
        if status != 'valid':
            failures.append(dict(image=image, reason=status, detail=detail))
        if i % 4000 == 0:
            print(f'  {i}/{len(images)} images checked', flush=True)
    return failures


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(connections=CONNECTIONS):
    images = split_images()
    print(f'{len(images)} unique images referenced by splits/', flush=True)

    missing = [image for image in images if not (ROOT / image).is_file()]
    if missing:
        print(f'{len(missing)} images absent; fetching the archive', flush=True)
        origin = download(connections=connections)
        print(f'archive {origin}: {ARCHIVE_PATH.relative_to(ROOT)}', flush=True)
        print(f'extracted {extract()} image files under {IMAGE_PREFIX}', flush=True)
    else:
        origin = 'cached'
        print('every referenced image is already on disk', flush=True)

    failures = validate(images)
    tables = ROOT / 'results/tables'
    tables.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed' if not failures else 'failed',
                  source_url=ARCHIVE_URL, wayback_snapshot=SNAPSHOT,
                  archive_bytes=ARCHIVE_BYTES,
                  archive_sha256=sha256(ARCHIVE_PATH) if ARCHIVE_PATH.is_file() else None,
                  archive_origin=origin, pillow_version=PIL.__version__,
                  checks=['snapshot_size_before_extraction', 'member_path_and_type',
                          'all_split_images_present', 'image_verify_and_full_decode'],
                  referenced_images=len(images), validated_images=len(images) - len(failures),
                  failures=failures)
    path = tables / 'image_fetch_report.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    require(not failures, f'{len(failures)} unusable images; see {path.relative_to(ROOT)}')
    print(f'{len(images)}/{len(images)} images valid. Report: {path.relative_to(ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch and validate the CrisisMMD images.')
    parser.add_argument('--connections', type=int, default=CONNECTIONS,
                        help='parallel range requests used to download the archive')
    main(parser.parse_args().connections)
