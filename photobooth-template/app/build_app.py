#!/usr/bin/env python3
"""Bundle editor.html and the 2x transparent template into one offline HTML file.

Usage: build_app.py OUTPUT.html [TEMPLATE.png]
"""
import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__TEMPLATE_DATA_URI__"


def main(out, template=HERE.parent / "template_transparent_2x.png"):
    html = (HERE / "editor.html").read_text(encoding="utf-8")
    assert html.count(PLACEHOLDER) == 1
    data = base64.b64encode(Path(template).read_bytes()).decode("ascii")
    Path(out).write_text(html.replace(PLACEHOLDER, "data:image/png;base64," + data), encoding="utf-8")
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
