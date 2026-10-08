"""Regenerate found_logicals.tex (TikZ panels) from the enumeration CSV. Run from the repo root."""
import csv, sys
rows = list(csv.DictReader(open("results/cosets/144_12_12_ldpc/enum_catalogue.csv")))
for r in rows: r["w"], r["n"] = int(r["weight"]), int(r["n_cosets"])
def panel(r):
    sup = [int(q) for q in r["support"].split()]
    nodes = []
    for q in sup:
        blk, rem = divmod(q, 72); a, b = divmod(rem, 12)
        nodes.append(f"\\node[{'Lq' if blk == 0 else 'Rq'}, minimum size=3pt] at \\{'L' if blk == 0 else 'R'}posT{{{a}}}{{{b}}} {{}};")
    return ("\\begin{tikzpicture}[scale=0.2]\n  \\torus\n  " + "\n  ".join(nodes) +
            f"\n  \\node[font=\\tiny, align=center] at (2.75,13.4) {{orbit {r['orbit']}\\\\ $w={r['w']}$, $\\times{r['n']}$}};\n\\end{{tikzpicture}}")
out = []
for w, k in ((12, 9), (14, 9)):
    sel = sorted([r for r in rows if r["w"] == w], key=lambda r: (-r["n"], int(r["orbit"])))[:k]
    out.append("\\par\\medskip\\noindent\n" + "\\hfill\n".join(panel(r) for r in sel) + "\n")
open("docs/logical_errors/found_logicals.tex", "w").write("% generated from results/cosets/144_12_12_ldpc/enum_catalogue.csv\n" + "".join(out))
print("ok")
