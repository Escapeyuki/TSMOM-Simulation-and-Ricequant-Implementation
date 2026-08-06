"""
Cut the paper's equations out of the PDF so the concordance can show them.

EQUATIONS.md and the dashboard both put the paper's typeset equation next to the
code that implements it. Retyping the maths in unicode or LaTeX would introduce a
third version that can drift from both; cropping the PDF cannot.

TWO WAYS TO FIND AN EQUATION, AND WHY THERE ARE TWO
    Numbered equations are located by their number: the text layer is searched
    for the token, and the crop is the height of that text block across the full
    column. That survives a re-paginated PDF.

    The catch is that this PDF's fonts carry a shifted encoding -- the glyphs
    that print as "(1)" come back from the text layer as the three characters
    "10Þ" -- so the anchors in eq_map.py are stored in that encoded form. If a
    different build of the paper ever decodes properly, the anchors stop
    matching, so every entry also carries an explicit rectangle to fall back to,
    and --verify refuses to accept a blank crop either way.

ENVIRONMENT
    Needs pymupdf, which in this project lives in /usr/local/bin/python3 (the rq
    conda env has rqdatac but not pymupdf). This script touches no market data,
    so it does not need the rq env.

RUN
    python make_eq_crops.py            # write docs/eq/*.png
    python make_eq_crops.py --verify   # check every crop is real
"""

import sys
from pathlib import Path

import fitz

from eq_map import CROPS, ENTRIES

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "Time series momentum - TimeSeriesMomentum.pdf"
DPI = 300

# A crop this uniform is a blank patch of paper: the anchor matched something
# unexpected, or the rectangle is off the text.
MIN_INK = 0.004          # fraction of pixels darker than mid-grey


def locate(page, entry):
    """The rectangle to cut. Returns (fitz.Rect, how) where how is the method."""
    anchor = entry.get("anchor")
    if anchor:
        hits = [w for w in page.get_text("words") if w[4] == anchor]
        if len(hits) == 1:
            x0, y0, x1, y1 = hits[0][:4]
            top, bottom = entry.get("pad", (8, 6))
            # Take the text block the number belongs to: that is the equation
            # line plus its superscripts and subscripts, which a bare word bbox
            # would clip.
            for blk in page.get_text("dict")["blocks"]:
                if blk.get("type") != 0:
                    continue
                bx0, by0, bx1, by1 = blk["bbox"]
                if bx0 - 2 <= x0 and x1 <= bx1 + 2 and by0 - 2 <= y0 and y1 <= by1 + 2:
                    y0, y1 = by0, by1
                    break
            col0, col1 = entry["col"]
            return fitz.Rect(col0, y0 - top, col1, y1 + bottom), "anchor"

    rect = entry.get("rect")
    if rect is None:
        return None, "none"
    return fitz.Rect(*rect), "rect"


def ink_fraction(pix):
    """Share of pixels darker than mid-grey. A blank crop scores ~0."""
    data = pix.samples
    n = pix.width * pix.height
    if not n:
        return 0.0
    stride = pix.n
    dark = sum(1 for i in range(0, len(data), stride * 7)     # sample, don't scan
               if data[i] < 128)
    return dark / max(1, n / 7)


def build(verify_only=False):
    doc = fitz.open(PDF)
    CROPS.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for entry in ENTRIES:
        if entry.get("page") is None:
            rows.append((entry["id"], "-", "no paper source", 0.0))
            continue

        page = doc[entry["page"] - 1]
        rect, how = locate(page, entry)
        if rect is None:
            failures.append(f"{entry['id']}: no anchor and no rect")
            continue

        pix = page.get_pixmap(clip=rect, dpi=DPI)
        ink = ink_fraction(pix)
        out = CROPS / f"{entry['id']}.png"
        if not verify_only:
            pix.save(out)
        rows.append((entry["id"], how, f"{pix.width}x{pix.height}", ink))
        if ink < MIN_INK:
            failures.append(f"{entry['id']}: crop is {ink:.4f} ink -- looks blank")
        if pix.width < 40 or pix.height < 12:
            failures.append(f"{entry['id']}: crop is {pix.width}x{pix.height} -- too small")

    print(f"{'id':<12}{'located by':<12}{'pixels':<14}{'ink':>8}")
    for eq_id, how, size, ink in rows:
        print(f"{eq_id:<12}{how:<12}{size:<14}{ink:>8.4f}")
    if not verify_only:
        print(f"\nwrote {len([r for r in rows if r[1] != '-'])} crops -> {CROPS}")

    if failures:
        print("\nPROBLEMS:")
        for f in failures:
            print(f"  {f}")
    return not failures


if __name__ == "__main__":
    ok = build(verify_only="--verify" in sys.argv)
    sys.exit(0 if ok else 1)
