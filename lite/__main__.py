import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .workflow import ROOT, build_graph


def main():
    parser = argparse.ArgumentParser(description='Read direction, generate a hypothesis, then generate factors.')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/research.yaml')
    parser.add_argument('--evaluation', type=Path, help='Completed exploration evaluation.json for a revision round')
    parser.add_argument('--env-file', type=Path, help='Explicit dotenv path; key is never saved to run output')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs')
    parser.add_argument('--resume', type=Path, help='Resume an existing loop run using its saved config and responses')
    parser.add_argument('--loop', action='store_true', help='Run bounded generation/evaluation loop (up to configured max_rounds, capped at 30)')
    parser.add_argument('--dry-run', action='store_true', help='Save rendered prompts without API access')
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error('--env-file does not exist')
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.resume and (args.dry_run or args.evaluation):
        parser.error('--resume cannot be combined with --dry-run or --evaluation')
    run = args.resume.resolve() if args.resume else args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    if args.resume:
        if not (run/'config.json').is_file(): parser.error('Resume directory has no config.json')
    else:
        run.mkdir(parents=True, exist_ok=False)
    # Advisory process lock avoids two processes spending API calls on one run.
    import fcntl
    lock = (run/'.run.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: parser.error('This run is already active')

    def save(name, value):
        (run/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    try:
        if args.loop or args.resume:
            from .loop import run_loop
            run_loop(args.config.resolve(), run, args.evaluation, dry_run=args.dry_run,resume=bool(args.resume))
            return 0
        result = build_graph(persist=save, dry_run=args.dry_run).invoke(
            {'config_path': str(args.config.resolve()), **({'evaluation_path': str(args.evaluation.resolve())} if args.evaluation else {})})
        if 'proposal' in result:
            save('proposal.json', result['proposal'])
        save('status.json', {'status': 'dry_run' if args.dry_run else 'proposed',
                            'formula_validation': 'not_run', 'factor_calculation': 'not_run'})
    except Exception as error:
        # Do not serialize provider exception text/headers, which may include credentials.
        existing = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        save('status.json', {**existing, 'status': 'failed', 'error_type': type(error).__name__})
        print(f'Failed: {type(error).__name__}. Check config, credentials, API access and run artifacts: {run}', file=sys.stderr)
        return 1
    print(f'{"Prompt preview (no API call)" if args.dry_run else "Proposal saved"}: {run}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
