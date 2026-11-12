"""Build static documentation locally; optionally push it to gh-pages."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Push generated HTML to origin/gh-pages")
    args = parser.parse_args()
    if args.publish and git("status", "--porcelain"):
        parser.error("Commit source changes before publishing to identify the build's source revision")
    revision = git("rev-parse", "HEAD")
    # A fresh temporary build cannot retain deleted pages or old result files.
    with tempfile.TemporaryDirectory(prefix="grasp-pages-") as temporary:
        base = Path(temporary)
        html = base/"html"
        subprocess.run([sys.executable, "-m", "sphinx", "-W", "-b", "html", str(ROOT/"docs"), str(html)],
                       cwd=ROOT, check=True)
        shutil.rmtree(html/".doctrees", ignore_errors=True)
        (html/".buildinfo").unlink(missing_ok=True)
        (html/".nojekyll").touch()
        (html/"source-revision.txt").write_text(revision+"\n")
        if not args.publish:
            output = ROOT/"docs/_build/html"
            if output.exists():
                shutil.rmtree(output)
            shutil.copytree(html, output)
            print(f"Static site built at {output}")
            return
        remote = git("remote", "get-url", "origin")
        pages = base/"pages"
        if git("ls-remote", "--heads", "origin", "refs/heads/gh-pages"):
            git("clone", "--single-branch", "--branch", "gh-pages", "--depth", "1", remote, str(pages))
            for path in pages.iterdir():
                if path.name not in {".git", "CNAME"}:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        else:
            pages.mkdir()
            git("init", "-b", "gh-pages", cwd=pages)
            git("remote", "add", "origin", remote, cwd=pages)
        shutil.copytree(html, pages, dirs_exist_ok=True)
        for key in ("user.name", "user.email"):
            git("config", key, git("config", key), cwd=pages)
        git("add", "--all", cwd=pages)
        if not git("status", "--porcelain", cwd=pages):
            print("Published HTML is already current")
            return
        git("commit", "-m", f"Publish documentation from {revision[:12]}", cwd=pages)
        git("push", "origin", "HEAD:refs/heads/gh-pages", cwd=pages)
        print(f"Published static HTML from {revision[:12]} to origin/gh-pages")


if __name__ == "__main__":
    main()
