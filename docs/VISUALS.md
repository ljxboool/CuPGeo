# Figure sources

All repository illustrations are original SVG artwork distributed with the source under the [MIT License](../LICENSE). The framework follows the manuscript's overview-and-details layout: a shared network flow above three module panels, with serif typography, three-dimensional feature blocks, heatmaps, and fine directional arrows. Input and mask thumbnails are synthetic vector illustrations. The figures use system fonts and self-contained vector shapes, with no external image dependencies.

## Editable masters

| File | Purpose |
| :--- | :--- |
| [cupgeo-method.svg](../assets/cupgeo-method.svg) | Shared representation, VRA, CP, and source-mask moment-ratio supervision |
| [cupgeo-protocol.svg](../assets/cupgeo-protocol.svg) | Matched CuPGeo training, independent continuations, frozen evaluation, and aggregation |
| [cupgeo-mark.svg](../assets/cupgeo-mark.svg) | Nested disc/cup identity at the top of each README |
| `assets/icon-*.svg` | Method, commands, protocol, results, code map, release, and citation navigation |

The method figure connects the shared representation to VRA, CP, the thresholded nested output, and training-only soft-vCDR supervision. Three lower panels expand the modules using allocation bars, vertical heatmaps, probability thumbnails, sigmoid curves, a residual-completion symbol, and probability-moment glyphs. Its editable groups are `method-overview`, `detail-vra`, `detail-cp`, `detail-ratio`, and `method-legend`. Dashed frames separate the panels; dotted links connect them to the overview. A two-row legend explains the glyphs and connection types. The portrait version places a compact overview above the same three detail panels.

The 1600 × 430 protocol master separates the source split, matched training, target evaluation, and reporting in `protocol-source`, `protocol-training`, `protocol-targets`, and `protocol-reporting`. It shares the method figure's serif typography, dashed section frames, and restrained gold, blue, green, and purple module colors. Its portrait version derives the experiment labels, budgets, counts, and seeds from the master. This figure covers the matched CuPGeo experiments; the separate MixStyle/DSU training budget remains documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Themes and narrow screens

Both READMEs use HTML `picture` sources for light/dark themes and a portrait layout at viewport widths up to 640 px. Full-size and portrait links also provide explicit access to the artwork. The small icons adapt through an internal color-scheme media query. This follows GitHub's documented [theme-aware image format](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github#adding-an-image-to-suit-your-visitors).

The method figure uses gold for the frozen backbone, blue-gray for the decoder, blue for VRA, green for CP, and purple for soft-vCDR. Text labels accompany the colors. Gray arrows indicate forward computation, blue arrows indicate prediction, orange dashed arrows indicate supervision, and pale dotted links identify module details. Times-style serif text and italic mathematical labels follow the manuscript's figure style. Light variants use a white canvas; dark variants use neutral dark backgrounds and surfaces while preserving the module colors. Titles and descriptions are embedded in each SVG; the README images also have descriptive alternative text.

## Update workflow

1. Edit the two master SVGs, preserving their named module groups and protocol data IDs.
2. Regenerate the six theme/portrait variants with `python3 -m scripts.build_diagram_variants`.
3. Run `python3 -m scripts.build_diagram_variants --check` to detect stale generated files.
4. Preview desktop and narrow layouts in both themes. Inspect text bounds, equation subscripts, branch arrows, detail links, heatmaps, and legend contrast against each theme's canvas.
5. Refresh the release's `SHA256SUMS` after the final edits and run `python3 -m scripts.check_release`.

The generator uses Python's standard library. It does not load datasets, model weights, or experiment outputs. GitHub Actions checks that generated variants match their source artwork.

When changing scientific annotations, check the model and experiment protocol as well as the drawing: VRA refines cup logits before CP; soft-vCDR is derived from final probability-map moments; and the five training continuations independently share the same-seed CP initialization.
