"""Build paper-style README SVG portraits and dark themes from editable masters.

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
FONT = "Times New Roman, Times, serif"


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


def canvas(source, height, title, width=480):
    root = ET.Element(f"{{{NS}}}svg", {"width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}", "role": "img", "aria-labelledby": "title desc"})
    element(root, "title", {"id": "title"}, title)
    element(root, "desc", {"id": "desc"}, source.find(f"{{{NS}}}desc").text)
    definitions = copy.deepcopy(source.find(f"{{{NS}}}defs"))
    if not any(node.get("id") == "arrow" for node in definitions.iter()):
        marker = copy.deepcopy(group_by_id(source, "forward"))
        marker.set("id", "arrow")
        definitions.append(marker)
    root.append(definitions)
    element(root, "rect", {"width": width, "height": height, "fill": "#FFFFFF", "id": "canvas"})
    return root, element(root, "g", {"font-family": FONT})


def block(parent, x, y, w, h, lines, fill="#FBE4D5", size=19):
    element(parent, "rect", {"x": x, "y": y, "width": w, "height": h, "rx": 2, "fill": fill})
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
    root, g = canvas(source, 1598, "CuPGeo — architecture and module details, portrait layout", width=510)
    label(g, 20, 35, "CuPGeo", 29, **{"font-weight": 700, "font-style": "italic"})
    label(g, 490, 33, "Architecture & geometric supervision", 16, "#868B92", **{"text-anchor": "end"})
    element(g, "rect", {"x": 20, "y": 55, "width": 470, "height": 273, "fill": "#FBFDFF",
                        "stroke": "#AFCADB", "stroke-dasharray": "6 4"})
    block(g, 37, 75, 436, 46, ["DINOv3-L/16 + LoRA → Pyramid-FPN"], "url(#module-blue)", 20)
    # The compact overview uses the same inputs to segmentation, VRA and CP.
    arrow(g, "M132 124V158")
    arrow(g, "M377 124V158")
    block(g, 37, 162, 194, 46, ["Segmentation head"], "url(#module-blue)", 20)
    block(g, 279, 162, 194, 46, ["VRA"], "url(#module-blue)", 21)
    arrow(g, "M234 185H275")
    label(g, 255, 176, "Z꜀", 17, **{"text-anchor": "middle"})
    arrow(g, "M132 211V264H150")
    arrow(g, "M377 211V264H359")
    block(g, 155, 241, 200, 46, ["CP composition"], "url(#module-green)", 22)
    label(g, 255, 312, "Shared threshold 0.5 → OC ⊆ OD", 20, **{"text-anchor": "middle"})
    for name, x, y in (("detail-vra", 20, 350), ("detail-cp", 12, 754), ("detail-ratio", 13, 1158)):
        panel = copy.deepcopy(group_by_id(source, name))
        panel.set("transform", f"translate({x} {y})")
        g.append(panel)
    label(g, 255, 1575, "Source training · frozen target inference", 19, "#868B92", **{"text-anchor": "middle"})
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
    block(g, 24, 98, 196, 66, ["REFUGE", f"{training_count} training images"], "url(#protocol-source-fill)", 19)
    block(g, 252, 98, 204, 66, ["CP pretraining", pretraining], "url(#protocol-pretrain-fill)", 19)
    arrow(g, "M223 131H248")
    arrow(g, "M354 168V203H50V449")
    label(g, 240, 194, "Same-seed source-selected CP best", 16, "#868B92", **{"text-anchor": "middle"})
    for i, name in enumerate(branches):
        y = 229+i*50
        element(g, "rect", {"x": 85, "y": y, "width": 371, "height": 39, "rx": 2, "fill": "url(#protocol-branch-fill)"})
        label(g, 101, y+26, name, 18, **{"data-contrast": "on-pastel", "font-weight": 600 if name == "Full CuPGeo" else 400})
        label(g, 440, y+25, continuation, 16, **{"text-anchor": "end", "data-contrast": "on-pastel"})
        arrow(g, f"M50 {y+20}H81")
    label(g, 240, 496, "Five independent continuations", 16, "#868B92", **{"text-anchor": "middle"})
    stage(g, 539, "Source-only selection")
    block(g, 24, 558, 432, 78, [f"REFUGE · {validation_count} validation images", "Best source-validation Mean Dice"], "url(#protocol-source-fill)", 20)
    stage(g, 681, "Frozen evaluation")
    for i, (name, count) in enumerate(zip(names, counts)):
        y = 701+i*43
        element(g, "rect", {"x": 24, "y": y, "width": 432, "height": 33, "rx": 2, "fill": "url(#protocol-target-fill)"})
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
    palette = {
        "#F5F5F5": "#242A32", "#FAFBFC": "#242A32", "#242424": "#E3E6EA", "#273440": "#E3E6EA",
        "#868B92": "#AFB9C4", "#536970": "#AFB9C4", "#C6CACF": "#596773",
        "#D5D8DC": "#47505B", "#B14A44": "#EDA092", "#647983": "#A1B3BF",
        "#657B86": "#A1B3BF", "#6C7D89": "#A1B3BF", "#C9D3DA": "#506171",
        "#FBFDFF": "#192630", "#FCFEFD": "#1C2B25", "#FDFCFF": "#282231",
        "#AFCADB": "#557186", "#ADCDBC": "#547966", "#C7B2DB": "#79618E",
        "#A4BED0": "#557186", "#A4C7B5": "#547966", "#C0AED3": "#79618E",
        "#C87855": "#DB9978",
    }
    for node in root.iter():
        # Keep scientific white probabilities white; recolor only canvas surfaces.
        if node.get("id") in {"canvas", "overview-surface"}:
            node.set("fill", "#141C24")
        elif node.tag.endswith("rect") and node.get("width") == "1600" and node.get("height") == "430":
            node.set("fill", "#141C24")
        for attr in ("fill", "stroke"):
            if node.get(attr) in palette:
                node.set(attr, palette[node.get(attr)])
        if node.get("data-contrast") == "on-pastel":
            node.set("fill", "#273440")
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
