"""Command line interface; reports stay on the user's machine."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import webbrowser


def parser():
    p=argparse.ArgumentParser(prog='disc-morphology',description='DICOM → automatic candidate mask → B, D, H, D/B → local interactive HTML')
    p.add_argument('--version',action='version',version='disc-morphology 0.1.0')
    sub=p.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect',help='List DICOM series without running models')
    inspect.add_argument('input',type=Path)
    setup=sub.add_parser('setup',help='Prepare the pinned optional SpineSegPlus backend')
    setup.add_argument('--model-home',type=Path)
    setup.add_argument('--python',type=Path,help='Reuse an existing model Python with torch/nnunetv2')
    setup.add_argument('--source',type=Path,help='Clone the pinned source from an existing local Git checkout')
    setup.add_argument('--assets',type=Path,help='Read published weight archives from a local directory')
    run=sub.add_parser('analyze',help='Analyze a directory containing one DICOM study')
    run.add_argument('input',type=Path)
    run.add_argument('-o','--output',type=Path)
    run.add_argument('--model-home',type=Path)
    run.add_argument('--sagittal-series',help='Explicit SeriesInstanceUID when automatic T2 selection is ambiguous')
    run.add_argument('--axial-series',help='Explicit axial SeriesInstanceUID')
    run.add_argument('--allow-missing-modality',action='store_true',help='Declare that anonymized DICOM without Modality is MR')
    run.add_argument('--allow-unverified-registration',action='store_true',help='Show a research axial projection despite differing/missing FrameOfReferenceUID')
    run.add_argument('--candidate',type=Path,help='Use an external binary NIfTI mask instead of running models')
    run.add_argument('--anatomy',type=Path,help='TotalSpineSeg label map corresponding to --candidate')
    run.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run.add_argument('--threads',type=int,default=2)
    run.add_argument('--timeout',type=int,default=3600,help='Inference timeout in seconds')
    run.add_argument('--no-open',action='store_true',help='Write the report without opening a browser')
    demo=sub.add_parser('demo',help='Generate a clearly labeled synthetic report; no models needed')
    demo.add_argument('-o','--output',type=Path)
    demo.add_argument('--no-open',action='store_true')
    verify=sub.add_parser('verify',help='Verify saved source/output SHA-256; does not rerun inference')
    verify.add_argument('directory',type=Path)
    return p


def open_report(path,no_open):
    if not no_open:
        if not webbrowser.open(Path(path).resolve().as_uri()):
            print('Browser could not be opened automatically. Open: '+str(path),file=sys.stderr)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and not argv[0].startswith('-') and argv[0] not in ('analyze','inspect','setup','demo','verify'):
        argv.insert(0,'analyze')
    p=parser();args=p.parse_args(argv)
    try:
        if args.command=='inspect':
            from .dicom import scan,series_info
            groups,ignored=scan(args.input)
            result={'series':series_info(groups),'ignored_nonimage_files':ignored}
        elif args.command=='setup':
            from .backend import default_home,setup
            home=args.model_home or default_home()
            setup(home,runtime_python=args.python,source=args.source,assets=args.assets)
            result={'status':'configured','model_home':str(Path(home).expanduser().resolve())}
        elif args.command=='verify':
            from .pipeline import verify
            result=verify(args.directory)
        else:
            suffix=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            output=args.output or Path.cwd()/('disc-'+('demo' if args.command=='demo' else 'report')+'-'+suffix)
            if args.command=='demo':
                from .demo import build_demo
                report=build_demo(output)
                result={'report':str(report),'synthetic':True,'inference_performed':False}
            else:
                from .backend import default_home
                from .pipeline import analyze
                if not 1<=args.threads<=16 or args.timeout<1:raise ValueError('threads must be 1–16 and timeout must be positive')
                if bool(args.candidate)!=bool(args.anatomy):raise ValueError('--candidate and --anatomy must be supplied together')
                report,data=analyze(args.input,output,model_home=args.model_home or default_home(),
                                    candidate_path=args.candidate,anatomy_path=args.anatomy,
                                    sagittal_series=args.sagittal_series,axial_series=args.axial_series,
                                    allow_missing_modality=args.allow_missing_modality,
                                    allow_unverified_registration=args.allow_unverified_registration,
                                    device=args.device,threads=args.threads,timeout=args.timeout)
                result={'report':str(report),'result':str(report.parent/'result.json'),
                        'inference_performed':data['model']['inference_performed'],
                        'clinical_classification':'indeterminate'}
            open_report(report,args.no_open)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,KeyError,RuntimeError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        print(json.dumps({'status':'error','message':str(e)},ensure_ascii=False),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted; no completed diagnosis was produced.',file=sys.stderr)
        return 130


if __name__=='__main__':raise SystemExit(main())
