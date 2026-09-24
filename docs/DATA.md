# Data preparation

Obtain the original datasets separately. This repository supplies converters and manifest schemas; it does not redistribute fundus images, masks, clinical labels, or the private manifests used in the original runs.

## Five-centre archive: REFUGE, BinRushed, Magrabia

The archived converter expects `Processed_Fundus_Images.zip`, Zenodo record **8009107**, with archive MD5 `a28baa241c45c95d14f9279e4c664d9a`. Extract it so that the official `<Domain>_train.csv` and `<Domain>_test.csv` files and their referenced images/masks are directly below `data/fundus_dg/extracted/`.

```bash
python -m scripts.prepare_fundus_dg_5centre \
  --dataset-root data/fundus_dg --manifest-dir manifests/fundus_dg

python -m scripts.build_fundus_dg_protocol \
  --manifest-dir manifests/fundus_dg --sources REFUGE --target BinRushed \
  --output-dir manifests/refuge_protocol

cp manifests/refuge_protocol/source_train.csv manifests/source_train.csv
cp manifests/refuge_protocol/source_val.csv manifests/source_val.csv
```

The protocol builder uses REFUGE's official `train` (320) and `test` (80) partitions as source train/validation. Its target is excluded from source fitting. For evaluation use `manifests/fundus_dg/binrushed_test.csv` (39) and `magrabia_test.csv` (19), with `--data-root data/fundus_dg`. The archive also contains ORIGA/Drishti-GS, which are not among the four reported targets in the current manuscript.

The raw mask encoding is 255=background, 128=disc rim, 0=cup. The converter writes canonical masks with 0=background, 1=rim, 2=cup. OD is the union of classes 1 and 2. For RIGA domains it resolves the archived expert-1 convention. Do not treat the raw 0/128/255 masks as canonical 0/1/2 masks.

## RIM-ONE DL

Place the separately obtained official images, reference masks, and license file under an extraction root, then:

```bash
python -m scripts.prepare_rim_one --archive-root data/raw/rim_one \
  --output-root data/rim_one --manifest-dir manifests
```

Use `manifests/rim_one_eval.csv` with `--data-root data/rim_one`: this is the official hospital-partition test set (174 images). The converter also creates other manifests for its original general-purpose workflow; CuPGeo training uses REFUGE manifests only.

## PAPILA

Place the official extracted PAPILA archive below an extraction root, then:

```bash
python -m scripts.prepare_papila --archive-root data/raw/papila \
  --output-root data/papila --manifest-dir manifests \
  --expert 1 --suspect-policy exclude --val-fraction 0.2 --seed 42
```

The converter rasterizes expert-1 contours and creates a deterministic patient-level split after excluding suspect diagnoses. Use `manifests/papila_eval.csv` with `--data-root data/papila`. The paper evaluated 84 images from this split, not the whole PAPILA collection. Check generated counts and metadata against this setting. The separate PAPILA training manifest is not a CuPGeo source-training set.

These conversion commands have been checked for their CLI interfaces during packaging, but have not been rerun on the full source archives. The original run manifests remain outside this package.

## CSV schema

For separate OD/OC binary masks, a labeled manifest has:

```csv
image,od_mask,oc_mask,glaucoma,patient_id,device,split,image_id,mask_encoding
images/example.png,masks/example_od.png,masks/example_oc.png,,example-patient,REFUGE,train,example,separate_binary
```

For canonical three-class masks, replace `od_mask,oc_mask` with `mask` and set `mask_encoding=three_class`. Keep the `glaucoma` column empty for datasets without diagnosis labels; the segmentation configs set classification loss weight to zero. Relative image/mask paths resolve against `--data-root`, not against the CSV location. IDs must be unique. Separate source patients between training and validation when patient identity is available; the five-centre archive's image-ID proxy is not evidence of patient-level disjointness.

```bash
python -m scripts.validate_manifest --csv manifests/source_train.csv \
  --mode source --check-paths --root data/fundus_dg
```

For image-only inference, a manifest can contain only `image,patient_id,device,split,image_id`. `scripts.evaluate --predictions ...` uses the image-only loader even if the supplied CSV also has labels. `score_predictions` separately needs the labeled CSV with matching image IDs.
