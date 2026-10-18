#!/usr/bin/env python3
"""Check local Markdown links and reject orphaned artifact documents."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse
from collections import deque


ROOT = pathlib.Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


class DocumentationError(RuntimeError):
    pass


def is_upstream_document(path: pathlib.Path) -> bool:
    """A document of a third-party crate, kept as its upstream ships it."""
    parts = path.relative_to(ROOT).parts
    return "third_party" in parts and parts[-2] != "third_party"


def markdown_files() -> set[pathlib.Path]:
    return {
        path.resolve()
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts
        and "results" not in path.parts
        and not is_upstream_document(path)
    }


def slug(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[^\w\- ]", "", value)
    value = re.sub(r"\s+", "-", value)
    return value


def anchors(path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def local_links(path: pathlib.Path) -> list[tuple[pathlib.Path, str | None]]:
    text = path.read_text(encoding="utf-8")
    result = []
    for raw in LINK.findall(text):
        decoded = urllib.parse.unquote(raw.strip("<>"))
        parsed = urllib.parse.urlsplit(decoded)
        if parsed.scheme or parsed.netloc:
            continue
        target_text = parsed.path
        if not target_text:
            target = path
        else:
            target = (path.parent / target_text).resolve()
        result.append((target, parsed.fragment or None))
    return result


def check() -> dict[str, int]:
    entry = (ROOT / "README.md").resolve()
    if not entry.is_file():
        raise DocumentationError("missing authoritative root README.md")
    all_docs = markdown_files()
    queue = deque([entry])
    reached: set[pathlib.Path] = set()
    checked_links = 0
    errors: list[str] = []
    while queue:
        document = queue.popleft()
        if document in reached:
            continue
        reached.add(document)
        for target, fragment in local_links(document):
            checked_links += 1
            try:
                target.relative_to(ROOT)
            except ValueError:
                errors.append(f"{document.relative_to(ROOT)}: link escapes repository")
                continue
            if not target.exists():
                errors.append(
                    f"{document.relative_to(ROOT)}: missing link target "
                    f"{target.relative_to(ROOT)}"
                )
                continue
            if fragment and target.is_file() and target.suffix.lower() == ".md":
                if fragment not in anchors(target):
                    errors.append(
                        f"{document.relative_to(ROOT)}: missing anchor "
                        f"#{fragment} in {target.relative_to(ROOT)}"
                    )
            if target.is_file() and target.suffix.lower() == ".md":
                queue.append(target)
    orphans = all_docs - reached
    if orphans:
        errors.append(
            "orphaned Markdown documents: "
            + ", ".join(str(path.relative_to(ROOT)) for path in sorted(orphans))
        )
    root_docs = sorted(
        path.name for path in ROOT.glob("*.md") if path.resolve() != entry
    )
    if root_docs:
        errors.append(
            "README.md must be the only root Markdown document: "
            + ", ".join(root_docs)
        )
    if errors:
        raise DocumentationError("\n".join(errors))
    return {
        "documents": len(all_docs),
        "reachable_documents": len(reached),
        "checked_links": checked_links,
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = check()
    print(
        "documentation checks passed: "
        f"{report['documents']} documents, {report['checked_links']} local links"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DocumentationError as exc:
        print(f"documentation check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
