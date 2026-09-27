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


def label(parent, x, y, text, size=16, fill="#173D43", **extra):
    return element(parent, "text", {"x": x, "y": y, "font-size": size, "fill": fill, **extra}, text)


def group_by_id(root, name):
    for node in root.iter():
        if node.get("id") == name:
            return node
    raise ValueError(f"Missing required SVG element: {name}")


def canvas(source, height, title):
    root = ET.Element(f"{{{NS}}}svg", {
        "width": "480", "height": str(height), "viewBox": f"0 0 480 {height}",
        "role": "img", "aria-labelledby": "title desc",
    })
    element(root, "title", {"id": "title"}, title)
    element(root, "desc", {"id": "desc"}, source.find(f"{{{NS}}}desc").text)
    root.append(copy.deepcopy(source.find(f"{{{NS}}}defs")))
    element(root, "rect", {"x": 1, "y": 1, "width": 478, "height": height - 2,
                           "rx": 20, "fill": "url(#paper)", "stroke": "#D9E2D9"})
    return root, element(root, "g", {"font-family": FONT})


def method_portrait(source):
    root, content = canvas(source, 1810, "CuPGeo method — portrait layout")
    label(content, 28, 37, "CuPGeo / METHOD OVERVIEW", 12, "#54766E", **{"letter-spacing": 1.6, "font-weight": 700})
    label(content, 28, 72, "Cup-preserving", 30, **{"font-weight": 700})
    label(content, 28, 106, "nested geometry", 30, **{"font-weight": 700})
    scale = 424 / 360
    for name, old_x, y in (("stage-shared", 28, 130), ("stage-vra", 420, 554), ("stage-cp", 812, 978)):
        stage = element(content, "g", {"transform": f"translate(28 {y}) scale({scale:.8f}) translate({-old_x} -148)"})
        stage.append(copy.deepcopy(group_by_id(source, name)))
    for y in (539, 963):
        element(content, "path", {"d": f"M240 {y}v9", "fill": "none", "stroke": "#72938C",
                                  "stroke-width": 1.5, "marker-end": "url(#arrow)"})
    for y, height in ((1404, 180), (1604, 164)):
        element(content, "rect", {"x": 28, "y": y, "width": 424, "height": height, "rx": 16,
                                  "fill": "url(#ink)", "stroke": "#416A64", "stroke-width": 0.6})
    # Retain the original equations and annotations. The decorative marginal
    # probability plot is omitted in portrait to keep its labels readable.
    ratio = element(content, "g", {"transform": "translate(-246 888)"})
    for node in group_by_id(source, "source-ratio"):
        if node.tag == f"{{{NS}}}text":
            ratio.append(copy.deepcopy(node))
    forward = element(content, "g", {"transform": "translate(-735 1088)"})
    forward.append(copy.deepcopy(group_by_id(source, "forward-geometry")))
    label(content, 240, 1793, "Source training · frozen target inference", 14, "#4B6E61", **{"text-anchor": "middle"})
    return root


def protocol_portrait(source):
    root, content = canvas(source, 1110, "CuPGeo experiment protocol — portrait layout")
    label(content, 26, 36, "CuPGeo / EXPERIMENT PROTOCOL", 11, "#54766E", **{"letter-spacing": 1.4, "font-weight": 700})
    label(content, 26, 71, "One source. Four targets.", 28, **{"font-weight": 700})
    total = group_by_id(source, "total-budget").text.split()[0]
    pretraining = group_by_id(source, "pretraining-budget").text
    continuation = group_by_id(source, "continuation-budget").text.removesuffix(" each")
    seeds = ", ".join(node.text for node in group_by_id(source, "seed-labels"))
    label(content, 26, 98, f"Matched {total}-epoch CuPGeo experiments", 16, "#54736A")

    def box(y, height, fill="url(#card)"):
        return element(content, "rect", {"x": 24, "y": y, "width": 432, "height": height,
                                         "rx": 14, "fill": fill, "stroke": "#DCE5DC"})

    box(124, 136)
    label(content, 44, 151, "01 / REFUGE SOURCE", 12, "#54766E", **{"font-weight": 700, "letter-spacing": 1})
    label(content, 44, 196, group_by_id(source, "source-train-count").text, 34, **{"font-weight": 700})
    label(content, 256, 196, group_by_id(source, "source-validation-count").text, 34, "#337F80", **{"font-weight": 700})
    label(content, 44, 227, "Training images", 16, "#54736A")
    label(content, 256, 227, "Source validation", 16, "#54736A")
    element(content, "path", {"d": "M233 169v62", "stroke": "#DCE5DC"})

    box(284, 320)
    label(content, 44, 313, "02 / MATCHED TRAINING", 12, "#54766E", **{"font-weight": 700, "letter-spacing": 1})
    label(content, 44, 349, f"CP pretraining · {pretraining}", 21, **{"font-weight": 700})
    label(content, 44, 375, "Same-seed CP best → independent branches", 15, "#54736A")
    # Read branch names from the master figure, so labels stay synchronized.
    branches = [node.text for node in group_by_id(source, "branch-labels")]
    if len(branches) != 5:
        raise ValueError("Expected five continuation labels in the protocol master")
    for i, name in enumerate(branches):
        y = 397 + i * 36
        element(content, "rect", {"x": 68, "y": y, "width": 366, "height": 29, "rx": 7,
                                  "fill": "url(#full)" if name == "Full CuPGeo" else "#FBFCF9", "stroke": "#DBE4D9"})
        label(content, 83, y + 20, name, 16, "#365E55", **{"font-weight": 700 if name == "Full CuPGeo" else 400})
        label(content, 419, y + 20, continuation, 14, "#966D4D", **{"text-anchor": "end"})
        element(content, "path", {"d": f"M48 {y+14}h15", "fill": "none", "stroke": "#72938C", "marker-end": "url(#arrow)"})
    element(content, "path", {"d": "M48 411v144", "stroke": "#A5BDB1"})
    label(content, 44, 591, "Select by source-validation Mean Dice", 14, "#4F7168")

    box(628, 224)
    label(content, 44, 657, "03 / FROZEN EVALUATION", 12, "#54766E", **{"font-weight": 700, "letter-spacing": 1})
    # Keep the benchmark sizes in the master SVG as the source of truth.
    names, counts = group_by_id(source, "target-labels"), group_by_id(source, "target-counts")
    if len(names) != 4 or len(counts) != 4:
        raise ValueError("Expected four domain names and counts in the protocol master")
    for i, (name_node, count_node) in enumerate(zip(names, counts)):
        row_y = 695 + i * 34
        label(content, 44, row_y, name_node.text, 18, "#365E55")
        label(content, 434, row_y, f"{count_node.text} images", 16, "#966D4D", **{"text-anchor": "end"})
    label(content, 44, 834, "Targets are used for evaluation only", 14, "#4F7168")

    box(876, 176, "url(#ink)")
    label(content, 44, 906, f"REPORTING / SEEDS {seeds}", 12, "#ADD0BE", **{"letter-spacing": 1.1, "font-weight": 700})
    label(content, 44, 944, "Equal four-domain average", 21, "#E5EEE3", **{"font-weight": 700})
    label(content, 44, 972, "within each seed", 16, "#B8CEC0")
    label(content, 44, 1010, "↓ Mean ± sample SD across seeds", 18, "#E5EEE3")
    for y in (266, 610, 858):
        element(content, "path", {"d": f"M240 {y}v10", "fill": "none", "stroke": "#72938C",
                                  "stroke-width": 1.5, "marker-end": "url(#arrow)"})
    label(content, 240, 1086, "One fixed source split · no target selection", 14, "#4B6E61", **{"text-anchor": "middle"})
    return root


def dark_variant(source):
    root = copy.deepcopy(source)
    # Change surfaces and text; preserve semantic OD/OC colors in the artwork.
    surfaces = {"paper": ("#152527", "#101C20"), "card": ("#213334", "#1A2B2D"),
                "full": ("#284841", "#213A36")}
    for node in root.iter():
        if node.get("id") in surfaces and node.tag.endswith("linearGradient"):
            for stop, color in zip(node, surfaces[node.get("id")]):
                stop.set("stop-color", color)
        if node.tag.endswith("feDropShadow"):
            node.set("flood-color", "#000000")
            node.set("flood-opacity", "0.18")
    text_colors = {
        "#173D43": "#E6EEE8", "#54766E": "#A8C7BB", "#60766F": "#B1C4B8",
        "#5D7B70": "#A7C3B6", "#5C766A": "#AEC5B8", "#597365": "#ADC5B7",
        "#3D6A61": "#B6D1C0", "#336B60": "#A6D2BF", "#4B6E61": "#B6CCBB",
        "#657F6F": "#A8BCAE", "#69847B": "#A6C4B8", "#54736A": "#BED1C3",
        "#337F80": "#A7D2C5", "#6C8278": "#A8C2B3", "#6B8378": "#A8C2B3",
        "#52746A": "#B2CCBE", "#4F7168": "#B6CCBE", "#7D8E81": "#AEBDAF",
        "#476E62": "#BCD6C7", "#966D4D": "#DFB58E", "#365E55": "#CADCCF",
        "#174B46": "#D6EADC", "#567E6B": "#C5DECA", "#809084": "#B4C3B5",
        "#947154": "#DCB997", "#78897E": "#B2C4B6",
    }
    strokes = {"#D9E2D9": "#354C47", "#D6E3D8": "#2D443E", "#DCE5DC": "#3B514A",
               "#E3E9DF": "#3D5048", "#E0E8DE": "#3D5048", "#C6D9CE": "#476354",
               "#DBE4D9": "#3F574C", "#ACCBBB": "#668C72"}
    for node in root.iter():
        if node.tag == f"{{{NS}}}text":
            # A text element can inherit fill from an enclosing group.
            for child in node.iter():
                if child.get("fill") in text_colors:
                    child.set("fill", text_colors[child.get("fill")])
        elif node.tag == f"{{{NS}}}g" and node.get("fill") in text_colors:
            if all(child.tag == f"{{{NS}}}text" for child in node):
                node.set("fill", text_colors[node.get("fill")])
        if node.get("stroke") in strokes:
            node.set("stroke", strokes[node.get("stroke")])
        if node.get("fill") == "#FBFCF9":
            node.set("fill", "#213631")
        if node.tag == f"{{{NS}}}ellipse" and node.get("fill") in {"#D9E6DD", "#D7E4D8"}:
            node.set("fill", "#061416")
    return root


def serialized(root):
    ET.indent(root, space="  ")
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
