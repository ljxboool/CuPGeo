# Figure sources

All repository illustrations are original SVG artwork distributed with the source under the [MIT License](../LICENSE). They contain no retinal photographs, dataset samples, model predictions, embedded fonts, or external image dependencies.

## Editable masters

| File | Purpose |
| :--- | :--- |
| [cupgeo-method.svg](../assets/cupgeo-method.svg) | Shared representation, VRA, CP, and source-mask moment-ratio supervision |
| [cupgeo-protocol.svg](../assets/cupgeo-protocol.svg) | Matched CuPGeo training, independent continuations, frozen evaluation, and aggregation |
| [cupgeo-mark.svg](../assets/cupgeo-mark.svg) | Nested disc/cup identity at the top of each README |
| `assets/icon-*.svg` | Method, commands, protocol, results, code map, release, and citation navigation |

The method figure has named groups for its three forward stages, source ratio objective, and forward geometry. Its portrait version rearranges those groups and retains the equations directly from the master. The portrait view omits the illustrative probability-profile plot to keep text readable. The protocol is for the matched CuPGeo experiments; the separate MixStyle/DSU training budget remains documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Themes and narrow screens

Both READMEs use HTML `picture` sources for light/dark themes and a portrait layout at viewport widths up to 640 px. Full-size and portrait links also provide explicit access to the artwork. The small icons adapt through an internal color-scheme media query. This follows GitHub's documented [theme-aware image format](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github#adding-an-image-to-suit-your-visitors).

Semantic color assignments remain fixed: teal for disc/rim and warm copper for cup. The dark variants adjust surfaces and text contrast while retaining those assignments. Text labels accompany the colors. Titles and descriptions are embedded in each SVG; the README images also have descriptive alternative text.

## Update workflow

1. Edit the two master SVGs, preserving the named method groups.
2. Regenerate the six theme/portrait variants with `python3 -m scripts.build_diagram_variants`.
3. Run `python3 -m scripts.build_diagram_variants --check` to detect stale generated files.
4. Preview desktop and narrow layouts in both themes. Inspect text bounds, equation subscripts, branch arrows, and small icon strokes.
5. Refresh the release's `SHA256SUMS` after the final edits and run `python3 -m scripts.check_release`.

The generator uses Python's standard library. It does not load datasets, model weights, or experiment outputs. GitHub Actions checks that generated variants match their source artwork.

When changing scientific annotations, check the model and experiment protocol as well as the drawing: VRA refines cup logits before CP; soft-vCDR is derived from final probability-map moments; and the five training continuations independently share the same-seed CP initialization.
