"""Build README theme/portrait variants from the editable SVG artwork.

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


def label(parent, x, y, text, size=16, fill="#273440", **extra):
    return element(parent, "text", {"x": x, "y": y, "font-size": size, "fill": fill, **extra}, text)


def group_by_id(root, name):
    for node in root.iter():
        if node.get("id") == name:
            return node
    raise ValueError(f"Missing required SVG element: {name}")


def canvas(source, height, title, width=480):
    root = ET.Element(f"{{{NS}}}svg", {
        "width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}",
        "role": "img", "aria-labelledby": "title desc",
    })
    element(root, "title", {"id": "title"}, title)
    element(root, "desc", {"id": "desc"}, source.find(f"{{{NS}}}desc").text)
    root.append(copy.deepcopy(source.find(f"{{{NS}}}defs")))
    element(root, "rect", {"width": width, "height": height, "fill": "#FFFFFF"})
    return root, element(root, "g", {"font-family": FONT})


def method_portrait(source):
    root, content = canvas(source, 1892, "CuPGeo — architecture and module details, portrait layout", width=510)
    label(content, 24, 39, "CuPGeo", 30, **{"font-family": "Georgia, Times New Roman, serif", "font-style": "italic", "font-weight": 700})
    label(content, 490, 37, "Architecture & module details", 16, "#677681", **{"text-anchor": "end"})
    element(content, "rect", {"x": 20, "y": 62, "width": 470, "height": 300,
                              "fill": "#FCFDFE", "stroke": "#BFCFDD", "stroke-dasharray": "7 5", "rx": 3})

    def box(x, y, w, h, title, fill, stroke, size=18):
        element(content, "rect", {"x": x, "y": y, "width": w, "height": h,
                                  "fill": fill, "stroke": stroke, "rx": 3})
        label(content, x+w/2, y+h/2+6, title, size, **{"text-anchor": "middle"})

    def arrow(d, prediction=False):
        kind = "prediction" if prediction else "forward"
        element(content, "path", {"d": d, "fill": "none", "stroke": "#2383D1" if prediction else "#6C7D89",
                                  "stroke-width": 1.8, "marker-end": f"url(#{kind})"})

    box(42, 86, 426, 52, "Input → DINOv3 / LoRA → Pyramid-FPN", "#FFF8E5", "#D9C585", 18)
    arrow("M137 140V167")
    arrow("M373 140V167")
    box(42, 172, 191, 45, "Segmentation head", "#E6EFF7", "#8BA9C0")
    box(277, 172, 191, 45, "VRA", "#E0EEF9", "#8EAFCB")
    arrow("M137 220V266H167")
    arrow("M373 220V266H343")
    box(172, 242, 166, 48, "CP composition", "#E0EFE6", "#8BB9A4")
    arrow("M255 293V313", prediction=True)
    label(content, 255, 338, "OD / OC · shared threshold 0.5", 18, **{"text-anchor": "middle"})
    # Copy complete panels rather than reconstruct their equations or arrows.
    for name, y in (("detail-vra", 388), ("detail-cp", 876), ("detail-ratio", 1364)):
        panel = copy.deepcopy(group_by_id(source, name))
        panel.set("transform", f"translate(20 {y})")
        content.append(panel)
    label(content, 255, 1860, "Source training · frozen target inference", 18, "#677681", **{"text-anchor": "middle"})
    return root


def protocol_portrait(source):
    root, content = canvas(source, 1110, "CuPGeo experiment protocol — portrait layout")
    label(content, 26, 36, "CuPGeo / EXPERIMENT PROTOCOL", 11, "#597285", **{"letter-spacing": 1.4, "font-weight": 700})
    label(content, 26, 71, "One source. Four targets.", 28, **{"font-weight": 700})
    total = group_by_id(source, "total-budget").text.split()[0]
    pretraining = group_by_id(source, "pretraining-budget").text
    continuation = group_by_id(source, "continuation-budget").text.removesuffix(" each")
    seeds = ", ".join(node.text for node in group_by_id(source, "seed-labels"))
    label(content, 26, 98, f"Matched {total}-epoch CuPGeo experiments", 16, "#677681")

    def box(y, height, fill="#FAFCFE"):
        return element(content, "rect", {"x": 24, "y": y, "width": 432, "height": height,
                                         "rx": 2, "fill": fill, "stroke": "#BFCFDD", "stroke-dasharray": "7 5"})

    box(124, 136)
    label(content, 44, 151, "01 / REFUGE SOURCE", 12, "#597285", **{"font-weight": 700, "letter-spacing": 1})
    label(content, 44, 196, group_by_id(source, "source-train-count").text, 34, **{"font-weight": 700})
    label(content, 256, 196, group_by_id(source, "source-validation-count").text, 34, "#496D8A", **{"font-weight": 700})
    label(content, 44, 227, "Training images", 16, "#677681")
    label(content, 256, 227, "Source validation", 16, "#677681")
    element(content, "path", {"d": "M233 169v62", "stroke": "#BFCFDD"})

    box(284, 320)
    label(content, 44, 313, "02 / MATCHED TRAINING", 12, "#597285", **{"font-weight": 700, "letter-spacing": 1})
    label(content, 44, 349, f"CP pretraining · {pretraining}", 21, **{"font-weight": 700})
    label(content, 44, 375, "Same-seed CP best → independent branches", 15, "#677681")
    # Read branch names from the master figure, so labels stay synchronized.
    branches = [node.text for node in group_by_id(source, "branch-labels")]
    if len(branches) != 5:
        raise ValueError("Expected five continuation labels in the protocol master")
    for i, name in enumerate(branches):
        y = 397 + i * 36
        element(content, "rect", {"x": 68, "y": y, "width": 366, "height": 29, "rx": 2,
                                  "fill": "url(#cp-fill)" if name == "Full CuPGeo" else "#FFFFFF", "stroke": "#CADDD2"})
        label(content, 83, y + 20, name, 16, "#426D58", **{"font-weight": 700 if name == "Full CuPGeo" else 400})
        label(content, 419, y + 20, continuation, 14, "#786889", **{"text-anchor": "end"})
        element(content, "path", {"d": f"M48 {y+14}h15", "fill": "none", "stroke": "#6C7D89", "marker-end": "url(#arrow)"})
    element(content, "path", {"d": "M48 411v144", "stroke": "#AACDBB"})
    label(content, 44, 591, "Select by source-validation Mean Dice", 14, "#5F7D6D")

    box(628, 224)
    label(content, 44, 657, "03 / FROZEN EVALUATION", 12, "#597285", **{"font-weight": 700, "letter-spacing": 1})
    # Keep the benchmark sizes in the master SVG as the source of truth.
    names, counts = group_by_id(source, "target-labels"), group_by_id(source, "target-counts")
    if len(names) != 4 or len(counts) != 4:
        raise ValueError("Expected four domain names and counts in the protocol master")
    for i, (name_node, count_node) in enumerate(zip(names, counts)):
        row_y = 695 + i * 34
        label(content, 44, row_y, name_node.text, 18, "#426D58")
        label(content, 434, row_y, f"{count_node.text} images", 16, "#786889", **{"text-anchor": "end"})
    label(content, 44, 834, "Targets are used for evaluation only", 14, "#5F7D6D")

    box(876, 176, "#F5F0FA")
    label(content, 44, 906, f"REPORTING / SEEDS {seeds}", 12, "#786889", **{"letter-spacing": 1.1, "font-weight": 700})
    label(content, 44, 944, "Equal four-domain average", 21, "#28363D", **{"font-weight": 700})
    label(content, 44, 972, "within each seed", 16, "#6D7280")
    label(content, 44, 1010, "↓ Mean ± sample SD across seeds", 18, "#28363D")
    for y in (266, 610, 858):
        element(content, "path", {"d": f"M240 {y}v10", "fill": "none", "stroke": "#6C7D89",
                                  "stroke-width": 1.5, "marker-end": "url(#arrow)"})
    label(content, 240, 1086, "One fixed source split · no target selection", 14, "#677681", **{"text-anchor": "middle"})
    return root


def dark_variant(source):
    # Retain the manuscript's white canvas and semantic palette in both themes.
    # A probability map or a scientific arrow must not change meaning with the UI.
    return copy.deepcopy(source)


def serialized(root):
    ET.indent(root, space="  ")
    # Indentation inside SVG text/tspan elements becomes visible word spacing.
    # Keep math subscripts adjacent to their symbols after pretty-printing.
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
