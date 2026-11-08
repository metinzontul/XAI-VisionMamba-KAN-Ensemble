"""
Author: Prof. Dr. Metin Zontul
Project: XAI-VisionMamba-KAN-Ensemble
File: figio.py

Common helper of the three figure scripts: every figure panel of the manuscript is written as its own file
(600 dpi PNG; plots additionally as vector PDF) into  <paper_dir>/article  or  <paper_dir>/supplementary.
No panel letters are drawn on the figures; the letter (a, b, c, ...) is part of the FILE NAME, so the panels can be
assembled and lettered in the manuscript (LaTeX subfigures).

File names:  Fig<number><letter>_<content>.png     e.g.  Fig3a_confusion_vim_s.png
             FigS<number><letter>_<content>.png    (supplementary)
"""

import os

LETTERS = "abcdefghijklmnop"
PAPER_DPI = 600


def paper_dirs(base):
    art, sup = os.path.join(base, "article"), os.path.join(base, "supplementary")
    os.makedirs(art, exist_ok=True)
    os.makedirs(sup, exist_ok=True)
    return art, sup


def save_panel(fig, folder, name, dpi=PAPER_DPI, pdf=False, tight=True):
    """PNG at `dpi` (dpi is also stored in the file); optional vector PDF. Returns the PNG path."""
    kw = dict(bbox_inches="tight") if tight else {}
    png = os.path.join(folder, name + ".png")
    fig.savefig(png, dpi=dpi, **kw)
    if pdf:
        fig.savefig(os.path.join(folder, name + ".pdf"), **kw)
    print(f"    wrote {os.path.relpath(png)}")
    return png
