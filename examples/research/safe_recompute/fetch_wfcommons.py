"""Загрузка зафиксированного корпуса WfCommons (171 трасса) с проверкой sha256.

Трассы (~309 МБ) в репозиторий не входят; их содержимое зафиксировано в
`data/wfcommons_manifest.json`. Файлы берутся из репозитория
github.com/wfcommons/wfinstances по тем же путям, что в
`planning/scripts/wfcommons_validation.py` (`<семейство>/[<подкаталог>/]<имя>.json`,
raw.githubusercontent.com), и сверяются с манифестом. Если трасса в WfInstances
изменилась после фиксации корпуса, загрузка завершается ошибкой: нужную ревизию
можно указать через --ref (git_blob_sha1 в манифесте позволяет найти её в истории).

Запуск:  python -m examples.research.safe_recompute.fetch_wfcommons [--ref main] [--dest DIR]
         python -m examples.research.safe_recompute.fetch_wfcommons verify [--dest DIR]
         python -m examples.research.safe_recompute.fetch_wfcommons manifest   # сопровождающему
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from examples.research.safe_recompute.common import (PLANNING_CACHE, WFCOMMONS_MANIFEST, WFCOMMONS_RAW,
                                                     _wfcommons_corpus, load_json, parse_wfformat,
                                                     save_json, sha256_of, wfcommons_manifest)

DEST = PLANNING_CACHE / "wfcommons"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(ref: str = "main", dest: Path = DEST) -> int:
    bad = 0
    for key, m in wfcommons_manifest().items():
        path = Path(dest) / m["path"]
        if path.exists() and sha256_of(path) == m["sha256"]:
            continue
        url = WFCOMMONS_RAW.format(ref=ref, path=m["path"])
        data = _download(url)
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            print(f"MISMATCH {key}: {url} differs from the pinned corpus", file=sys.stderr)
            bad += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"fetched {key}")
    return bad


def verify(dest: Path = DEST) -> int:
    bad = 0
    for key, m in wfcommons_manifest().items():
        path = Path(dest) / m["path"]
        if not path.exists() or sha256_of(path) != m["sha256"]:
            print(f"{'MISSING' if not path.exists() else 'MISMATCH'} {key}", file=sys.stderr)
            bad += 1
    print(f"{len(wfcommons_manifest()) - bad} ok, {bad} bad")
    return bad


def build_manifest(src: Path = DEST) -> dict:
    """Манифест по локальному кэшу: только трассы корпуса, порядок обхода -- семейства
    nextflow, snakemake, pegasus, внутри -- отсортированные пути (как раньше в `wfcommons_graphs`)."""
    corpus = _wfcommons_corpus()
    files: dict[str, dict] = {}
    for fam in ("nextflow", "snakemake", "pegasus"):
        for path in sorted((Path(src) / fam).rglob("*.json")):
            if (fam, path.stem) not in corpus:
                continue
            key = f"{fam}/{path.stem}"
            if key in files:
                raise ValueError(f"duplicate key {key}")
            raw = path.read_bytes()
            g = parse_wfformat(path)
            files[key] = {"path": path.relative_to(src).as_posix(),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "bytes": len(raw),
                          "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(),
                          "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges()}
    if len(files) != len(corpus):
        raise ValueError(f"corpus has {len(corpus)} traces, found {len(files)} in {src}")
    return {"source": {"repository": "https://github.com/wfcommons/wfinstances",
                       "url_template": WFCOMMONS_RAW,
                       "corpus": "examples/research/planning/data/wfcommons_validation_results.json",
                       "traces": len(files), "bytes": sum(f["bytes"] for f in files.values())},
            "files": files}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="fetch", choices=["fetch", "verify", "manifest"])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--dest", default=str(DEST))
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        save_json(build_manifest(Path(a.dest)), WFCOMMONS_MANIFEST)
        print(f"{WFCOMMONS_MANIFEST}: {load_json(WFCOMMONS_MANIFEST)['source']['traces']} traces")
        return
    bad = fetch(a.ref, Path(a.dest)) if a.cmd == "fetch" else verify(Path(a.dest))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
