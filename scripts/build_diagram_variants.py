"""Build compact README SVG portraits and dark themes from editable masters.

Standard library only; no datasets, model dependencies, or raster conversion.
Run after editing assets/cupgeo-method.svg or assets/cupgeo-protocol.svg.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
FONT = "Arial, Helvetica, sans-serif"


def element(parent, tag, attrs=None, text=None):
    node = ET.SubElement(parent, f"{{{NS}}}{tag}", {k: str(v) for k, v in (attrs or {}).items()})
    node.text = text
    return node


def label(parent, x, y, text, size=18, fill="#242424", **extra):
    node = element(parent, "text", {"x": x, "y": y, "font-size": size, "fill": fill, **extra}, text)
    # Use actual SVG subscripts rather than relying on Unicode small-cap glyphs.
    symbols = {"Z꜀": ("Z", "C"), "Zᵣ": ("Z", "R"), "P꜀": ("P", "C")}
    for token, (base, sub) in symbols.items():
        if token in text:
            before, after = text.split(token, 1)
            node.text = before + base
            element(node, "tspan", {"baseline-shift": "sub", "font-size": "70%"}, sub).tail = after
            break
    return node


def group_by_id(root, name):
    for node in root.iter():
        if node.get("id") == name:
            return node
    raise ValueError(f"Missing required SVG element: {name}")


def canvas(source, height, title):
    root = ET.Element(f"{{{NS}}}svg", {"width": "480", "height": str(height),
        "viewBox": f"0 0 480 {height}", "role": "img", "aria-labelledby": "title desc"})
    element(root, "title", {"id": "title"}, title)
    element(root, "desc", {"id": "desc"}, source.find(f"{{{NS}}}desc").text)
    root.append(copy.deepcopy(source.find(f"{{{NS}}}defs")))
    element(root, "rect", {"width": 480, "height": height, "fill": "#FFFFFF", "id": "canvas"})
    return root, element(root, "g", {"font-family": FONT})


def block(parent, x, y, w, h, lines, fill="#FBE4D5", size=19):
    element(parent, "rect", {"x": x, "y": y, "width": w, "height": h, "rx": 9, "fill": fill})
    for i, line in enumerate(lines):
        label(parent, x+w/2, y+h/2+(i-(len(lines)-1)/2)*24+6, line, size,
              **{"text-anchor": "middle", "data-contrast": "on-pastel"})


def arrow(parent, d, training=False):
    attrs = {"d": d, "fill": "none", "stroke": "#242424", "stroke-width": 1.5,
             "marker-end": "url(#arrow)"}
    if training:
        attrs["stroke-dasharray"] = "5 4"
    element(parent, "path", attrs)


def stage(parent, y, text):
    label(parent, 240, y, text, 16, "#868B92", **{"text-anchor": "middle"})
    half = len(text)*4.4 + 18
    element(parent, "path", {"d": f"M24 {y-5}H{240-half}M{240+half} {y-5}H456",
                              "fill": "none", "stroke": "#C6CACF", "stroke-width": 1.3})


def method_portrait(source):
    root, g = canvas(source, 850, "CuPGeo — compact architecture, portrait layout")
    label(g, 24, 36, "CuPGeo", 25, **{"font-weight": 600})
    label(g, 456, 34, "Source-only segmentation", 16, "#868B92", **{"text-anchor": "end"})
    stage(g, 76, "Representation")
    block(g, 24, 98, 202, 60, ["DINOv3-L/16"], "#DFECF8")
    block(g, 245, 98, 74, 60, ["LoRA"], size=17)
    block(g, 338, 98, 118, 60, ["Pyramid-FPN"], size=16)
    arrow(g, "M228 128H241")
    arrow(g, "M321 128H334")
    arrow(g, "M397 161V189H124V222")
    arrow(g, "M358 189V222")
    block(g, 30, 226, 184, 58, ["Segmentation head"], size=18)
    block(g, 266, 226, 184, 58, ["VRA"], size=20)
    arrow(g, "M217 253H262")
    label(g, 239, 243, "Z꜀", 16, "#B14A44", **{"text-anchor": "middle"})
    arrow(g, "M124 288V368H141")
    arrow(g, "M358 288V368H339")
    label(g, 116, 326, "Z꜀", 18, "#B14A44", **{"text-anchor": "end"})
    label(g, 371, 326, "B", 18, "#B14A44")
    block(g, 145, 338, 190, 60, ["Cup refine", "σ(Z꜀ + B)"], "#E2EFD9")
    arrow(g, "M240 402V448")
    label(g, 252, 428, "P꜀", 18, "#B14A44")
    arrow(g, "M27 254H16V490H141")
    label(g, 29, 425, "Zᵣ", 18, "#B14A44")
    block(g, 145, 452, 190, 76, ["CP composition", "SG on disc path"], "#E2EFD9", 18)
    arrow(g, "M240 532V562")
    block(g, 67, 566, 346, 60, ["Threshold 0.5 → OC ⊆ OD"], "#E2EFD9", 20)
    arrow(g, "M243 548H440V676", training=True)
    label(g, 24, 661, "Training only", 16, "#868B92", **{"font-style": "italic"})
    block(g, 24, 680, 192, 56, ["Source-mask", "moments"], "#E2EFD9", 18)
    block(g, 250, 680, 206, 56, ["Soft-vCDR", "Smooth L1 · λ = 2"], "#E2EFD9", 18)
    arrow(g, "M219 708H246", training=True)
    # Reuse the exact composition from the master, including math subscripts.
    equation = copy.deepcopy(list(group_by_id(source, "method-legend"))[-1])
    equation.set("x", "240")
    equation.set("y", "776")
    equation.set("font-size", "21")
    equation.set("text-anchor", "middle")
    g.append(equation)
    element(g, "rect", {"x": 16, "y": 797, "width": 448, "height": 37, "rx": 6, "fill": "#F5F5F5"})
    for x, color, name in ((29, "#DFECF8", "frozen"), (164, "#FBE4D5", "trainable"), (320, "#E2EFD9", "geometry")):
        element(g, "rect", {"x": x, "y": 806, "width": 20, "height": 19, "rx": 3, "fill": color})
        label(g, x+28, 821, name, 16)
    return root


def protocol_portrait(source):
    root, g = canvas(source, 1040, "CuPGeo — matched experimental protocol, portrait layout")
    total = group_by_id(source, "total-budget").text.split()[0]
    pretraining = group_by_id(source, "pretraining-budget").text
    continuation = group_by_id(source, "continuation-budget").text.removesuffix(" each")
    branches = [node.text for node in group_by_id(source, "branch-labels")]
    names = [node.text for node in group_by_id(source, "target-labels")]
    counts = [node.text for node in group_by_id(source, "target-counts")]
    seeds = ", ".join(node.text for node in group_by_id(source, "seed-labels"))
    if len(branches) != 5 or len(names) != 4 or len(counts) != 4:
        raise ValueError("Expected five continuations and four target domains")
    label(g, 24, 37, "Matched experiments", 25, **{"font-weight": 600})
    label(g, 456, 35, f"{total} epochs", 17, "#868B92", **{"text-anchor": "end"})
    stage(g, 78, "Source training")
    training_count = group_by_id(source, "source-train-count").text
    validation_count = group_by_id(source, "source-validation-count").text
    block(g, 24, 98, 196, 66, ["REFUGE", f"{training_count} training images"], "#DFECF8", 19)
    block(g, 252, 98, 204, 66, ["CP pretraining", pretraining], size=19)
    arrow(g, "M223 131H248")
    arrow(g, "M354 168V203H50V449")
    label(g, 240, 194, "Same-seed source-selected CP best", 16, "#868B92", **{"text-anchor": "middle"})
    for i, name in enumerate(branches):
        y = 229+i*50
        element(g, "rect", {"x": 85, "y": y, "width": 371, "height": 39, "rx": 7, "fill": "#FBE4D5"})
        label(g, 101, y+26, name, 18, **{"data-contrast": "on-pastel", "font-weight": 600 if name == "Full CuPGeo" else 400})
        label(g, 440, y+25, continuation, 16, **{"text-anchor": "end", "data-contrast": "on-pastel"})
        arrow(g, f"M50 {y+20}H81")
    label(g, 240, 496, "Five independent continuations", 16, "#868B92", **{"text-anchor": "middle"})
    stage(g, 539, "Source-only selection")
    block(g, 24, 558, 432, 78, [f"REFUGE · {validation_count} validation images", "Best source-validation Mean Dice"], "#DFECF8", 20)
    stage(g, 681, "Frozen evaluation")
    for i, (name, count) in enumerate(zip(names, counts)):
        y = 701+i*43
        element(g, "rect", {"x": 24, "y": y, "width": 432, "height": 33, "rx": 6, "fill": "#E2EFD9"})
        label(g, 40, y+23, name, 18, **{"data-contrast": "on-pastel"})
        label(g, 440, y+23, f"{count} images", 16, **{"text-anchor": "end", "data-contrast": "on-pastel"})
    label(g, 240, 893, "Targets used for evaluation only", 16, "#868B92", **{"text-anchor": "middle"})
    element(g, "rect", {"x": 16, "y": 916, "width": 448, "height": 108, "rx": 6, "fill": "#F5F5F5"})
    label(g, 35, 945, f"Seeds {seeds}", 18)
    label(g, 35, 975, "Equal four-domain mean within each seed", 17)
    label(g, 35, 1004, "→ Mean ± sample SD across seeds", 17)
    return root


def dark_variant(source):
    root = copy.deepcopy(source)
    palette = {"#FFFFFF": "#171A1F", "#F5F5F5": "#24282F", "#242424": "#E3E6EA",
               "#868B92": "#A7AFBA", "#C6CACF": "#59616C", "#D5D8DC": "#47505B",
               "#B14A44": "#EDA092"}
    for node in root.iter():
        for attr in ("fill", "stroke"):
            if node.get(attr) in palette:
                node.set(attr, palette[node.get(attr)])
        if node.get("data-contrast") == "on-pastel":
            # Some labels inherit their fill from the source group.
            node.set("fill", "#242424")
    return root


def serialized(root):
    ET.indent(root, space="  ")
    for node in root.iter(f"{{{NS}}}text"):
        for part in node.iter():
            if part.text and part.text.isspace():
                part.text = None
            if part is not node and part.tail and part.tail.isspace():
                part.tail = None
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated variants are stale; do not write files.")
    args = parser.parse_args()
    expected = {}
    for name, portrait in (("method", method_portrait), ("protocol", protocol_portrait)):
        base = ET.parse(ROOT / "assets" / f"cupgeo-{name}.svg").getroot()
        mobile = portrait(base)
        expected[f"cupgeo-{name}-dark.svg"] = serialized(dark_variant(base))
        expected[f"cupgeo-{name}-mobile.svg"] = serialized(mobile)
        expected[f"cupgeo-{name}-mobile-dark.svg"] = serialized(dark_variant(mobile))
    stale = []
    for name, content in expected.items():
        path = ROOT / "assets" / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        else:
            path.write_text(content)
    if stale:
        parser.error("Stale diagram variants: " + ", ".join(stale))
    print(f"{'Verified' if args.check else 'Built'} {len(expected)} SVG variants")


if __name__ == "__main__":
    main()
