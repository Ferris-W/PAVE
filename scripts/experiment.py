"""Evaluate or train the full PAVE student and its full-context teacher."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('teacher', 'pave')
DATASETS = {
    'PEMS03': dict(keep_ratio=60 / 88, road_fraction=0.5, batch_size=16, query_chunk=0),
    'PEMS07': dict(keep_ratio=60 / 88, road_fraction=0., batch_size=16, query_chunk=0),
    'PEMS08': dict(keep_ratio=0.8, road_fraction=0., batch_size=16, query_chunk=0),
    'CHENGDU': dict(keep_ratio=60 / 88, road_fraction=0., batch_size=1, query_chunk=128),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def saved_checkpoint(args, role):
    folder = args.checkpoints_root.resolve() if args.checkpoints_root is not None else ROOT / 'checkpoints'
    return folder / args.dataset / f'{role}.pt'


def check_saved(path):
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')


def train_command(args, spec, role, output, teacher=None):
    is_teacher = role == 'teacher'
    config_dir = ROOT / 'configs' / 'teachers' if is_teacher else ROOT / 'configs'
    config = config_dir / f'{args.dataset}.json'
    cmd = [sys.executable, '-u', str(ROOT / 'model/train.py'), '--dataset', args.dataset,
           '--data-root', str(args.data_root), '--config', str(config),
           '--variant', 'relation_kd' if role == 'pave' else 'no_kd',
           '--seed', '0', '--node-seed', '0', '--node-sampling', 'stfit',
           '--split-ratio', '.6', '.2', '.2', '--node-ratio', '.1',
           '--target-channel', '0', '--in-steps', '12', '--out-steps', '12',
           '--device', args.device, '--max-epochs', '400', '--output', str(output)]
    if not is_teacher:
        cmd += ['--context-keep-ratio', str(spec['keep_ratio']),
                '--context-subset-probability', '1.0',
                '--context-road-fraction', str(spec['road_fraction'])]
    if role == 'pave':
        if teacher is None:
            raise ValueError('KD needs the full-context teacher checkpoint')
        cmd += ['--teacher-checkpoint', str(teacher)]
    if spec['query_chunk']:
        cmd += ['--eval-batch-size', str(spec['batch_size']),
                '--eval-query-chunk', str(spec['query_chunk'])]
    return cmd + (['--smoke'] if args.smoke else [])


def evaluate_command(args, spec, role, output):
    return [sys.executable, '-u', str(ROOT / 'model/evaluate.py'),
            '--checkpoint', str(saved_checkpoint(args, role)),
            '--data-root', str(args.data_root), '--device', args.device,
            '--batch-size', str(spec['batch_size']),
            '--query-chunk', str(spec['query_chunk']), '--output', str(output)]


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(output, rows):
    save_csv(output / 'comparison.csv', rows)
    effects = []
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        group = {r['role']: r for r in rows if r['dataset'] == dataset}
        if 'pave' not in group:
            continue
        for control in ('teacher',):
            if control not in group:
                continue
            row = dict(dataset=dataset, treatment='pave', control=control)
            for metric in ('mae', 'rmse', 'mape_percent'):
                a, b = group[control][metric], group['pave'][metric]
                row[metric + '_reduction'] = a - b
                row[metric + '_reduction_percent'] = (a - b) / a * 100 if a else None
            effects.append(row)
    save_csv(output / 'ablation.csv', effects)


def metric_row(dataset, role, metric, source):
    return dict(dataset=dataset, role=role, **metric, source=source)


def completed_run(parent, dataset, smoke):
    paths = list(parent.glob('*/metrics.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one completed result under {parent}')
    report = read_json(paths[0])
    if report.get('status') != 'complete' or report.get('smoke') != smoke or report['dataset'] != dataset:
        raise ValueError(f'Run is incomplete or belongs to another task: {paths[0]}')
    if not (paths[0].parent / 'best.pt').is_file():
        raise FileNotFoundError(paths[0].parent / 'best.pt')
    return paths[0].parent, report


def run_child(cmd, log):
    print('COMMAND:', shlex.join(cmd), flush=True)
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        stream.write('COMMAND: ' + shlex.join(cmd) + '\n'); stream.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    if code:
        print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-40:]), flush=True)
        raise RuntimeError(f'Child exited {code}: {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaluate', 'train'))
    parser.add_argument('--dataset', type=str.upper, choices=tuple(DATASETS), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data', help='Directory containing <DATASET>/<DATASET>.npz and road files')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--role', choices=('all', 'teacher', 'pave'), default='all',
                        help='Evaluate either model; train teacher only or the student with a saved teacher')
    parser.add_argument('--teacher-source', choices=('fresh', 'saved'), default='fresh',
                        help='Train: fresh trains a full teacher first; saved reuses the archived teacher')
    parser.add_argument('--checkpoints-root', type=Path, help='Optional relocated checkpoints bundle')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/reproduce')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Train only; short execution, excluded from formal tables')
    args = parser.parse_args()
    if args.smoke and args.mode != 'train':
        parser.error('--smoke is only supported for training')
    if args.mode == 'train' and args.role == 'teacher' and args.teacher_source == 'saved':
        parser.error('Teacher-only training requires --teacher-source fresh')
    if args.mode == 'train' and args.role == 'pave' and args.teacher_source != 'saved':
        parser.error('--role pave requires --teacher-source saved; use --role all to train both')
    roles = ROLES if args.role == 'all' else (args.role,)
    if args.data_root:
        args.data_root = args.data_root.expanduser().resolve()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    batch = args.output.resolve() / f"{args.mode}-{'smoke-' if args.smoke else ''}{args.dataset or 'all'}-{stamp}"
    spec = DATASETS[args.dataset]
    if args.dry_run:
        if args.mode == 'evaluate':
            commands = [evaluate_command(args, spec, role, batch / f'{role}.json') for role in roles]
        elif args.mode == 'train':
            teacher = saved_checkpoint(args, 'teacher') if args.teacher_source == 'saved' else batch / 'teacher/<new-run>/best.pt'
            stages = roles if args.teacher_source == 'fresh' else tuple(r for r in roles if r != 'teacher')
            commands = [train_command(args, spec, role, batch / role, teacher) for role in stages]
        else:
            commands = []
        for cmd in commands:
            print(shlex.join(cmd))
        return
    for name in (f'{args.dataset}.npz', f'{args.dataset}.csv', f'adj_{args.dataset}.pkl'):
        if not (args.data_root / args.dataset / name).is_file():
            raise FileNotFoundError(args.data_root / args.dataset / name)
    if args.mode == 'evaluate':
        for role in roles:
            check_saved(saved_checkpoint(args, role))
    elif args.mode == 'train' and args.teacher_source == 'saved':
        check_saved(saved_checkpoint(args, 'teacher'))
    batch.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', mode=args.mode, dataset=args.dataset, smoke=args.smoke,
                    teacher_source=args.teacher_source if args.mode == 'train' else None,
                    commands=[], completed={})
    rows = []
    print('Results:', batch, flush=True)
    def dispatch(role, cmd):
        manifest['commands'].append(dict(role=role, argv=cmd))
        write_json(batch / 'experiment.json', manifest)
        run_child(cmd, batch / f'{role}.log')
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.mode == 'evaluate':
            for role in roles:
                output = batch / f'{role}.json'
                dispatch(role, evaluate_command(args, spec, role, output))
                report = read_json(output)
                if report['dataset'] != args.dataset:
                    raise ValueError('Checkpoint belongs to a different dataset')
                rows.append(metric_row(args.dataset, role, report['metrics']['unseen']['prefix']['12'], str(output)))
                manifest['completed'][role] = str(output)
        elif args.mode == 'train':
            teacher = saved_checkpoint(args, 'teacher') if args.teacher_source == 'saved' else None
            for role in (roles if teacher is None else tuple(r for r in roles if r != 'teacher')):
                dispatch(role, train_command(args, spec, role, batch / role, teacher))
                folder, report = completed_run(batch / role, args.dataset, args.smoke)
                if role == 'teacher':
                    teacher = folder / 'best.pt'
                rows.append(metric_row(args.dataset, role, report['metrics']['unseen']['prefix']['12'], str(folder)))
                manifest['completed'][role] = str(folder)
        if not args.smoke:
            tables(batch, rows)
        manifest['status'] = 'complete'
        write_json(batch / 'experiment.json', manifest)
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        write_json(batch / 'experiment.json', manifest)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print('Completed:', batch, flush=True)


if __name__ == '__main__':
    main()
