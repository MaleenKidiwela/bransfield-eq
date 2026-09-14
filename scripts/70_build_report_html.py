"""Render notes/32_methods_results_draft.md into the HTML report published as the
'Orca Seismicity Catalogue' artifact (https://claude.ai/code/artifact/310e571d-5329-43ac-9696-e709a0b095e2).

Figure placeholders '**Figure N:** ...' become <figure> blocks; the N -> PNG mapping below
points at notes/figures/final/ (script 69). Placeholders without a PNG render as
'not yet produced' boxes with their specification. Output: <out>/index.html + <out>/fig/*.png;
publish index.html with the fig/ files as supporting files. Requires `pip install markdown`.
usage: PYTHONPATH=src python3 scripts/70_build_report_html.py --out /path/to/dir
"""
from __future__ import annotations
import argparse, html, re, shutil
from pathlib import Path
import markdown

REPO = Path(__file__).resolve().parent.parent
FIGMAP = {4: ["F6_traveltime_solver_check.png"], 5: ["F4_station_terms.png"], 6: ["F5_sp_gate.png"],
          10: ["F1_map_epicentres.png"], 11: ["F3_depth_distributions.png", "F2_depth_sections.png"],
          12: ["F7_temporal.png"], 13: ["F8_nlloc_vs_hypodd_depth.png"]}
CSS_PATH = REPO / "notes" / "report_style.css"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "notes" / "32_methods_results_draft.md"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "fig").mkdir(parents=True, exist_ok=True)
    md = Path(a.src).read_text()
    md = md.split("\n---\n", 1)[1] if "\n---\n" in md else md
    lines = md.split("\n"); res = []; i = 0
    while i < len(lines):
        m = re.match(r"\*\*Figure (\d+):\*\*\s*(.*)", lines[i])
        if m:
            n = int(m.group(1)); para = [m.group(2)]; i += 1
            while i < len(lines) and lines[i].strip(): para.append(lines[i].strip()); i += 1
            cap = html.escape(" ".join(para), quote=False)
            cap = re.sub(r"`([^`]*)`", r"<code>\1</code>", cap)
            imgs = FIGMAP.get(n, [])
            body = "".join(f'<img src="fig/{f}" alt="Figure {n}" loading="lazy">' for f in imgs) if imgs else \
                '<div class="fig-pending">Figure not yet produced — specification below</div>'
            res.append(f'<figure class="fig{"" if imgs else " pending"}" id="fig{n}">{body}<figcaption><span class="figno">Figure {n}</span> {cap}</figcaption></figure>')
            continue
        res.append(lines[i]); i += 1
    body = markdown.markdown("\n".join(res), extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
                             extension_configs={"toc": {"toc_depth": "1-2"}})
    body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    toc = "\n".join(f'<a class="l{l}" href="#{h}">{re.sub("<.*?>", "", t)}</a>'
                    for l, h, t in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', body))
    css = CSS_PATH.read_text()
    page = (REPO / "notes" / "report_template.html").read_text().format(css=css, toc=toc, body=body)
    (out / "index.html").write_text(page)
    for f in sum(FIGMAP.values(), []):
        shutil.copy(REPO / "notes" / "figures" / "final" / f, out / "fig" / f)
    print(f"wrote {out/'index.html'} ({len(page):,} bytes), {len(sum(FIGMAP.values(), []))} figures")


if __name__ == "__main__":
    main()
