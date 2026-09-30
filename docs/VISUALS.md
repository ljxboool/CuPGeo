# Figure sources

All repository illustrations are original SVG artwork distributed with the source under the [MIT License](../LICENSE). The framework and protocol use a compact block-flow layout: pastel modules, thin charcoal arrows, stage guides above the flow, and a shared legend strip below it. Input and mask thumbnails are synthetic vector illustrations, not retinal photographs, dataset samples, or measured model predictions. No embedded fonts or external image dependencies are required.

## Editable masters

| File | Purpose |
| :--- | :--- |
| [cupgeo-method.svg](../assets/cupgeo-method.svg) | Shared representation, VRA, CP, and source-mask moment-ratio supervision |
| [cupgeo-protocol.svg](../assets/cupgeo-protocol.svg) | Matched CuPGeo training, independent continuations, frozen evaluation, and aggregation |
| [cupgeo-mark.svg](../assets/cupgeo-mark.svg) | Nested disc/cup identity at the top of each README |
| `assets/icon-*.svg` | Method, commands, protocol, results, code map, release, and citation navigation |

The method figure reads from shared representation through cup geometry to nested prediction, with training objectives shown alongside the relevant outputs. Its editable groups are `shared-representation`, `cup-geometry`, `nested-prediction`, `training-objective`, and `method-legend`. The portrait version reflows the same stages into a compact vertical sequence. Stage guides and direct connections organize the drawing without enclosing each module in a separate detail panel.

The protocol master separates the source split, matched training, target evaluation, and reporting in `protocol-source`, `protocol-training`, `protocol-targets`, and `protocol-reporting`. Its portrait version derives the experiment labels, budgets, counts, and seeds from the master. This figure covers the matched CuPGeo experiments; the separate MixStyle/DSU training budget remains documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Themes and narrow screens

Both READMEs use HTML `picture` sources for light/dark themes and a portrait layout at viewport widths up to 640 px. Full-size and portrait links also provide explicit access to the artwork. The small icons adapt through an internal color-scheme media query. This follows GitHub's documented [theme-aware image format](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github#adding-an-image-to-suit-your-visitors).

Module colors identify function: light blue for the frozen backbone, peach for trainable modules, and green for fixed geometry and objectives. Text labels accompany the colors. Light variants use a white canvas; dark variants use neutral dark backgrounds and surfaces while preserving legible pastel module blocks. Titles and descriptions are embedded in each SVG; the README images also have descriptive alternative text.

## Update workflow

1. Edit the two master SVGs, preserving their named module groups and protocol data IDs.
2. Regenerate the six theme/portrait variants with `python3 -m scripts.build_diagram_variants`.
3. Run `python3 -m scripts.build_diagram_variants --check` to detect stale generated files.
4. Preview desktop and narrow layouts in both themes. Inspect text bounds, equation subscripts, branch arrows, stage guides, and legend contrast against each theme's canvas.
5. Refresh the release's `SHA256SUMS` after the final edits and run `python3 -m scripts.check_release`.

The generator uses Python's standard library. It does not load datasets, model weights, or experiment outputs. GitHub Actions checks that generated variants match their source artwork.

When changing scientific annotations, check the model and experiment protocol as well as the drawing: VRA refines cup logits before CP; soft-vCDR is derived from final probability-map moments; and the five training continuations independently share the same-seed CP initialization.
