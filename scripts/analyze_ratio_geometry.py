"""CPU-only geometry audit and FP32 probability export for frozen predictions.

Run --spec jobs.json --output DIR --source-root SOURCE --threads 2.
Each job: method, seed, domain, artifact, manifest, data_root; optional expected
checkpoint_sha256 and expected_manifest_sha256. No training or GPU operations.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, records):
    if not records:
        return
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(clean(records))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--reuse-probabilities', action='store_true',
                        help='Require each job probability_index; verify and reuse existing FP32 files without exporting.')
    parser.add_argument('--compute-proxy', action='store_true',
                        help='Compute moment targets and proxy statistics for this job set only.')
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 2:
        parser.error('CPU analysis is limited to at most two threads')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from scipy.stats import pearsonr, spearmanr
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, args.source_root)
    from c3tta.data.manifest import read_manifest
    from c3tta.engine.offline_evaluation import _load_segmentation_target
    from c3tta.metrics import dice_score
    if args.compute_proxy:
        from c3tta.losses.multitask import soft_vcdr_from_masks

    spec = json.loads(Path(args.spec).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def extent(mask):
        rows = np.flatnonzero(mask.any(axis=1))
        return int(rows[-1] - rows[0] + 1) if len(rows) else 0

    def avg(values):
        valid = [float(x) for x in values if x is not None and math.isfinite(float(x))]
        return float(np.mean(valid)) if valid else None

    def moments(values):
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return {'mean': avg(vals), 'sample_sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                'n': len(vals), 'by_seed': values}

    def corr(rows, a, b):
        pairs = [(r[a], r[b]) for r in rows if r[a] is not None and r[b] is not None]
        if len(pairs) < 2:
            return {'n': len(pairs), 'pearson': None, 'spearman': None}
        x, y = np.asarray(pairs).T
        return {'n': len(pairs),
                'pearson': float(pearsonr(x, y).statistic) if x.std() and y.std() else None,
                'spearman': float(spearmanr(x, y).statistic) if x.std() and y.std() else None}

    def pair_stats(rows, a, b):
        pairs = [(r[a], r[b]) for r in rows if r[a] is not None and r[b] is not None
                 and math.isfinite(float(r[a])) and math.isfinite(float(r[b]))]
        if not pairs:
            return {'n': 0, 'invalid': len(rows), 'pearson': None, 'spearman': None,
                    'mae': None, 'bias_first_minus_second': None}
        x, y = np.asarray(pairs).T
        return {**corr([{a: first, b: second} for first, second in pairs], a, b),
                'invalid': len(rows) - len(pairs), 'mae': float(np.abs(x-y).mean()),
                'bias_first_minus_second': float((x-y).mean()),
                'rmse': float(np.sqrt(np.mean((x-y)**2))),
                'max_absolute_gap': float(np.abs(x-y).max())}

    metrics = ['od_dice', 'oc_dice', 'mean_dice', 'hard_vcdr_absolute_error',
               'hard_vcdr_signed_error', 'cvr']
    for part in ('od', 'oc'):
        metrics += [part + suffix for suffix in (
            '_diameter_normalized_absolute_error', '_diameter_normalized_signed_error',
            '_diameter_same_grid_absolute_error_px', '_diameter_same_grid_signed_error_px')]
    records, jobs, gt_cache = [], [], {}
    for job in spec['jobs']:
        method, seed, domain = job['method'], int(job['seed']), job['domain']
        key = (str(job['manifest']), str(job['data_root']))
        artifact = torch.load(job['artifact'], map_location='cpu', weights_only=True, mmap=True)
        meta = artifact['metadata']
        if 'source_seed' in meta:
            assert int(meta['source_seed']) == seed
        manifest_sha = sha_file(job['manifest'])
        assert meta.get('inference_manifest_sha256', manifest_sha) == manifest_sha
        if job.get('expected_checkpoint_sha256'):
            assert meta['checkpoint_sha256'] == job['expected_checkpoint_sha256']
        if job.get('expected_manifest_sha256'):
            assert manifest_sha == job['expected_manifest_sha256']
        logits = artifact['predictions']['seg_logits']
        assert logits.ndim == 4 and logits.shape[1:] == (2, 768, 768), tuple(logits.shape)
        probability_index = None
        if args.reuse_probabilities:
            index_path = Path(job['probability_index']).resolve()
            probability_index = json.loads(index_path.read_text())
            assert probability_index['channels'] == ['OD', 'OC']
            assert probability_index['probability_dtype'] == 'float32'
            assert probability_index['source_sha256'] == sha_file(job['artifact'])
            assert [r['image_id'] for r in probability_index['records']] == artifact['image_ids']
        if key not in gt_cache:
            rows = read_manifest(job['manifest'])
            truth = []
            for row in rows:
                native = _load_segmentation_target(row, Path(job['data_root']), None).numpy() >= .5
                resized = _load_segmentation_target(row, Path(job['data_root']), 768)
                native_d = [extent(ch) for ch in native]
                resized_d = [extent(ch) for ch in (resized.numpy() >= .5)]
                entry = {'id': row.image_id, 'patient_id': row.patient_id,
                              'native_height': native.shape[1], 'native_width': native.shape[2],
                              'native_diameters': native_d, 'resized_diameters': resized_d,
                              'resized': resized.bool()}
                if args.compute_proxy:
                    entry['soft_native'] = float(soft_vcdr_from_masks(torch.from_numpy(native).unsqueeze(0)).item())
                    entry['soft_resized'] = float(soft_vcdr_from_masks(resized.unsqueeze(0)).item())
                truth.append(entry)
            gt_cache[key] = truth
        truth = gt_cache[key]
        assert artifact['image_ids'] == [t['id'] for t in truth]
        out = output / method / f'seed{seed}' / domain
        out.mkdir(parents=True, exist_ok=True)
        probs_dir = out / 'probabilities'
        if not args.reuse_probabilities:
            probs_dir.mkdir(parents=True, exist_ok=True)
        group = []
        stored_soft = artifact['predictions'].get('soft_vcdr')
        for i, target in enumerate(truth):
            prob = torch.sigmoid(logits[i].float())
            assert bool(torch.isfinite(prob).all())
            arr = prob.numpy()
            mask = arr >= .5
            pred_d = [extent(ch) for ch in mask]
            gt_d = target['native_diameters']
            ratio_valid = pred_d[0] > 0 and gt_d[0] > 0
            pred_ratio = pred_d[1] / pred_d[0] if pred_d[0] else None
            target_ratio = gt_d[1] / gt_d[0] if gt_d[0] else None
            ratio_error = pred_ratio - target_ratio if ratio_valid else None
            if probability_index is not None:
                old = probability_index['records'][i]
                probability_path = (index_path.parent / old['file']).resolve()
                assert probability_path.is_relative_to(index_path.parent)
                assert sha_file(probability_path) == old['sha256']
                saved = np.load(probability_path, mmap_mode='r', allow_pickle=False)
                assert list(saved.shape) == old['shape'] == [2, 768, 768]
                assert saved.dtype == np.float32 and np.array_equal(saved, arr)
            else:
                probability_path = probs_dir / (f'{i:04d}_' + hashlib.sha256(target['id'].encode()).hexdigest()[:16] + '.npy')
                temp = probability_path.with_suffix('.npy.tmp')
                with temp.open('wb') as stream:
                    np.save(stream, arr, allow_pickle=False)
                temp.replace(probability_path)
            dice = [float(dice_score(prob[j:j+1], target['resized'][j:j+1])[0]) for j in range(2)]
            row = {'method': method, 'seed': seed, 'domain': domain, 'image_id': target['id'],
                   'patient_id': target['patient_id'], 'od_dice': dice[0], 'oc_dice': dice[1],
                   'mean_dice': (dice[0] + dice[1]) / 2,
                   'cvr': int(np.any(mask[1] & ~mask[0])),
                   'hard_vcdr_prediction': pred_ratio, 'hard_vcdr_target': target_ratio,
                   'hard_vcdr_valid': int(ratio_valid),
                   'hard_vcdr_signed_error': ratio_error,
                   'hard_vcdr_absolute_error': abs(ratio_error) if ratio_error is not None else None,
                   'soft_vcdr_stored': float(stored_soft[i].item()) if stored_soft is not None else None,
                   'native_height': target['native_height'], 'native_width': target['native_width'],
                   'prediction_height': 768, 'prediction_width': 768,
                   'probability_path': str(probability_path), 'probability_dtype': str(arr.dtype),
                   'probability_file_sha256': sha_file(probability_path),
                   'source_logit_dtype': str(logits.dtype), 'checkpoint_sha256': meta.get('checkpoint_sha256'),
                   'prediction_artifact_path': job['artifact']}
            if args.compute_proxy:
                row.update(soft_vcdr_target_native_moment=target['soft_native'],
                           soft_vcdr_target_resized_moment=target['soft_resized'],
                           soft_vcdr_prediction_reconstructed=float(soft_vcdr_from_masks(prob.unsqueeze(0)).item()))
            for j, part in enumerate(('od', 'oc')):
                delta_norm = pred_d[j] / 768 - gt_d[j] / target['native_height']
                delta_px = pred_d[j] - target['resized_diameters'][j]
                row.update({part+'_prediction_nonempty': int(pred_d[j] > 0),
                            part+'_gt_native_nonempty': int(gt_d[j] > 0),
                            part+'_diameter_prediction_px': pred_d[j],
                            part+'_diameter_gt_native_px': gt_d[j],
                            part+'_diameter_gt_resized_px': target['resized_diameters'][j],
                            part+'_diameter_prediction_normalized': pred_d[j] / 768,
                            part+'_diameter_gt_native_normalized': gt_d[j] / target['native_height'],
                            part+'_diameter_normalized_signed_error': delta_norm,
                            part+'_diameter_normalized_absolute_error': abs(delta_norm),
                            part+'_diameter_same_grid_signed_error_px': delta_px,
                            part+'_diameter_same_grid_absolute_error_px': abs(delta_px)})
            group.append(row)
        summary = {'method': method, 'seed': seed, 'domain': domain, 'n_total': len(group),
                   'n_vcdr_valid': sum(r['hard_vcdr_valid'] for r in group),
                   'n_pred_od_nonempty': sum(r['od_prediction_nonempty'] for r in group),
                   'n_pred_oc_nonempty': sum(r['oc_prediction_nonempty'] for r in group),
                   'n_gt_od_nonempty': sum(r['od_gt_native_nonempty'] for r in group),
                   'n_gt_oc_nonempty': sum(r['oc_gt_native_nonempty'] for r in group),
                   'metrics': {m: avg([r[m] for r in group]) for m in metrics},
                   'manifest_sha256': manifest_sha, 'checkpoint_sha256': meta.get('checkpoint_sha256'),
                   'artifact_path': job['artifact'], 'artifact_bytes': Path(job['artifact']).stat().st_size,
                   'artifact_mtime_ns': Path(job['artifact']).stat().st_mtime_ns,
                   'logit_dtype': str(logits.dtype), 'probability_shape_per_image': [2, 768, 768],
                   'probability_dtype': 'float32', 'n_probability_files': len(group),
                   'checkpoint_epoch': meta.get('checkpoint_epoch')}
        summary['probability_files_reused'] = args.reuse_probabilities
        if args.compute_proxy:
            summary['proxy_statistics'] = {
                'soft_prediction_vs_hard_prediction': pair_stats(group, 'soft_vcdr_stored', 'hard_vcdr_prediction'),
                'soft_prediction_vs_moment_gt': pair_stats(group, 'soft_vcdr_stored', 'soft_vcdr_target_resized_moment'),
                'soft_prediction_vs_hard_gt': pair_stats(group, 'soft_vcdr_stored', 'hard_vcdr_target'),
                'native_moment_gt_vs_hard_gt': pair_stats(group, 'soft_vcdr_target_native_moment', 'hard_vcdr_target'),
                'resized_moment_gt_vs_hard_gt': pair_stats(group, 'soft_vcdr_target_resized_moment', 'hard_vcdr_target'),
                'reconstructed_soft_vs_stored': pair_stats(group, 'soft_vcdr_prediction_reconstructed', 'soft_vcdr_stored')}
        if job.get('expected_hard_vcdr_mae') is not None:
            error = abs(summary['metrics']['hard_vcdr_absolute_error'] - job['expected_hard_vcdr_mae'])
            assert error < 1e-7, (job, error)
            summary['hard_vcdr_mae_difference_from_existing_audit'] = error
        write_csv(out / 'per_image.csv', group)
        dump(out / 'summary.json', summary)
        records += group
        jobs.append(summary)
        del artifact, logits
        gc.collect()
        print(json.dumps({'completed_job': f'{method}/seed{seed}/{domain}', 'images': len(group),
                          'elapsed_seconds': time.time() - start}), flush=True)

    write_csv(output / 'per_image.csv', records)
    aggregates = {}
    target_domains = spec.get('target_domains', ['binrushed', 'magrabia', 'rim_one', 'papila'])
    for method in sorted(set(r['method'] for r in jobs)):
        seeds = sorted(set(r['seed'] for r in jobs if r['method'] == method))
        entry = {'by_domain': {}, 'four_target_macro': {}}
        for domain in sorted(set(r['domain'] for r in jobs)):
            selected = [r for seed in seeds for r in jobs if r['method'] == method and r['seed'] == seed and r['domain'] == domain]
            entry['by_domain'][domain] = {m: moments([r['metrics'][m] for r in selected]) for m in metrics}
        for m in metrics:
            by_seed = []
            for seed in seeds:
                selected = [r for r in jobs if r['method'] == method and r['seed'] == seed and r['domain'] in target_domains]
                assert len(selected) == len(target_domains)
                by_seed.append(avg([r['metrics'][m] for r in selected]))
            entry['four_target_macro'][m] = moments(by_seed)
        aggregates[method] = entry

    paired_rows, paired_summaries = [], []
    for pair in spec.get('pairs', []):
        before, after = pair['before'], pair['after']
        index = {(r['method'], r['seed'], r['domain'], r['image_id']): r for r in records}
        for a in records:
            if a['method'] != before:
                continue
            b = index[after, a['seed'], a['domain'], a['image_id']]
            row = {'comparison': after+' minus '+before, 'seed': a['seed'], 'domain': a['domain'], 'image_id': a['image_id']}
            for m in metrics:
                row['delta_'+m] = b[m] - a[m] if b[m] is not None and a[m] is not None else None
            paired_rows.append(row)
        for seed in sorted(set(r['seed'] for r in paired_rows)):
            for domain in sorted(set(r['domain'] for r in paired_rows)):
                rows = [r for r in paired_rows if r['seed'] == seed and r['domain'] == domain and r['comparison'] == after+' minus '+before]
                paired_summaries.append({'comparison': after+' minus '+before, 'seed': seed, 'domain': domain,
                    'n_paired': len(rows), 'mean_deltas': {m: avg([r['delta_'+m] for r in rows]) for m in metrics},
                    'delta_correlations': {
                        'od_dice_vs_od_normalized_diameter_absolute_error': corr(rows, 'delta_od_dice', 'delta_od_diameter_normalized_absolute_error'),
                        'od_diameter_absolute_error_vs_hard_vcdr_absolute_error': corr(rows, 'delta_od_diameter_normalized_absolute_error', 'delta_hard_vcdr_absolute_error'),
                        'oc_diameter_absolute_error_vs_hard_vcdr_absolute_error': corr(rows, 'delta_oc_diameter_normalized_absolute_error', 'delta_hard_vcdr_absolute_error')}})
    write_csv(output / 'paired_per_image.csv', paired_rows)
    dump(output / 'summary.json', {'schema': 'cupgeo.diameter_probability_audit.v1',
        'cpu_threads': args.threads, 'cuda_used': False, 'elapsed_seconds': time.time() - start,
        'jobs': jobs, 'aggregates': aggregates, 'paired_per_seed_domain': paired_summaries,
        'n_images': len(records), 'n_probability_files': len(records),
        'definitions': {
            'diameter': 'Inclusive vertical foreground extent; empty masks have diameter zero.',
            'primary_diameter_error': 'predicted extent/768 minus native GT extent/native height; absolute and signed errors include empty predictions.',
            'secondary_diameter_error': 'Predicted extent minus nearest-neighbor resized GT extent, both on 768 grid, in pixels.',
            'vCDR_valid': 'Predicted and ground-truth OD must be nonempty; empty predicted cup gives vCDR=0 and remains included.',
            'CVR': 'Image-level indicator of any OC foreground outside OD at threshold 0.5.',
            'probability_export': 'One .npy per image, [OD,OC] channels, shape 2x768x768, sigmoid(stored logits cast to FP32). These are FP32 files derived from previously quantized FP16 logits and do not recover unquantized network outputs.',
            'aggregation': 'Four target domains equally averaged within each seed; across-seed mean and sample SD (ddof=1). Source validation kept separate.',
            'pairing': 'Exact method/seed/domain/image_id; all cases retained, including PAPILA outliers.',
            'proxy_statistics': 'Per-job moment targets/statistics computed only for supplied jobs.' if args.compute_proxy else 'Existing soft_vcdr_evidence_20260921 statistics reused through supplied spec; no rerun of the 75-artifact proxy study.'},
        'reused_proxy_evidence': spec.get('reused_proxy_evidence')})


if __name__ == '__main__':
    main()
