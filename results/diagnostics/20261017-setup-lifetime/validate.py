from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
diag = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'scripts'))
import reproduce as r

manifest = r.build_manifest()
previous = root / 'results/runs/20261016T185738Z-full-7a99e5d775'
summary = {'publication_results': False, 'cpu': 2, 'jobs': []}
job_names = [
    'point-client-undine-k8', 'point-client-borg-k8',
    'point-client-undine-k16', 'point-client-borg-k16',
    'point-helper-undine-k32', 'point-helper-borg-k32',
    'range-helper-undine-k64', 'range-helper-borg-k64',
    'on-the-fly-undine-k8',
]
for name in job_names:
    job = manifest.jobs[name]
    point = name.startswith('point-')
    trials = 163 if point else 4
    env = r.clean_benchmark_environment()
    settings = job.env_dict() | {job.trial_env: str(trials)}
    if name.startswith('range-'):
        prefix = 'BENCH_APPEND_ONLY_LOG_RANGE_OPEN_'
        settings.update({prefix + 'BYTE_LEN_LIST': '1024,1048576',
                         prefix + 'COMMITTED_BULK_UPDATE_BATCHES': '1'})
    env.update(settings)
    directory = diag / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=True)
    binary = diag / 'binaries' / job.layout / job.benchmark
    command = ['taskset', '--cpu-list', '2', str(binary), '--bench']
    print(name, flush=True)
    record = {'id': name, 'command': command, 'settings': settings,
              'binary_sha256': r.sha256_file(binary)}
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        result = subprocess.run(command, cwd=directory, env=env,
                                stdout=out, stderr=err, timeout=180)
    record['exit_code'] = result.returncode
    assert result.returncode == 0, (name, result.returncode)
    output = directory / job.result_relpath
    document = json.loads(output.read_text())
    r.validate_timing_policy(r.RunJob(job, tuple(manifest.measurements_for_job(name))), document)
    record['result'] = str(output.relative_to(diag))
    record['result_sha256'] = r.sha256_file(output)
    record['entries'] = len(document['entries'])
    if point:
        old = json.loads((previous / 'jobs' / name / 'batch-000/result.json').read_text())
        def shape(entries):
            return [{key: value for key, value in entry.items() if not isinstance(value, dict)}
                    for entry in entries]
        assert shape(document['entries']) == shape(old['entries']), name
        record['shape_matches_previous'] = True
        metrics = manifest.measurements_for_job(name)
        samples, _, _, _ = r.extract_measurements(document, metrics, warmup_repetitions=3)
        record['timings'] = {key: r.summarize_samples(values) for key, values in samples.items()}
        for key, stats in record['timings'].items():
            print(key, 'n=', stats['n'], 'mean_us=', round(stats['mean']/1000, 3),
                  'relative_half_width=', round(stats['sequential_ci']['relative_half_width'], 4), flush=True)
    summary['jobs'].append(record)
    (diag / 'validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
summary['all_passed'] = True
(diag / 'validation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('All nine targeted benchmark jobs passed', flush=True)
