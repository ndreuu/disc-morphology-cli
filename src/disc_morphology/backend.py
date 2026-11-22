"""Install a pinned optional model backend and run it in its own interpreter."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tarfile
import urllib.request
import venv

from .dicom import sha256

SOURCE = 'https://github.com/Albertzry/SpineSegPlus.git'
COMMIT = '66fd963b4e7e4f9530e6cf7b983b23348712dcab'
ASSETS = {
    'Dataset101_TotalSpineSeg_step1.tar.gz':'826084a1837f53ea05936190e058012369e843d5726dad71ed160a41362a4264',
    'Dataset102_TotalSpineSeg_step2.tar.gz':'10a2e709d7984d4191f865255064815ea715675dd0b4d2104667711a23ebc666',
    'Dataset105_TotalSpineSeg_LDH.tar.gz':'e9e3c53d64f31f1ecd22634f783f15c7b522656b68971acd51a5203ccf5cba3a',
    'Dataset107_LDH_ROI.tar.gz':'334b236ce7b5243901ab791a911b44619e0db497b1b095eb0206510eeba319c0',
}


def default_home():
    configured=os.environ.get('DISC_MORPHOLOGY_MODEL_HOME')
    local=Path.cwd()/'.models'/'spinesegplus'
    fallback=local if (local/'config.json').is_file() else Path.home()/'.cache'/'disc-morphology'/'spinesegplus'
    return Path(configured or fallback).expanduser().resolve()


def _run(args, **kwargs):
    return subprocess.run([str(x) for x in args],check=True,**kwargs)


def safe_extract(archive, target, expected_prefix):
    """Only bounded regular checkpoint/JSON files, without links or traversal."""
    target = Path(target).resolve()
    total = 0
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            p = Path(member.name)
            if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=expected_prefix
                    or member.issym() or member.islnk()):
                raise ValueError('Unsafe model archive path/link')
            if member.isdir():continue
            if not member.isfile():raise ValueError('Unexpected model archive member')
            total += member.size
            if member.size > 3*1024**3 or total > 5*1024**3:
                raise ValueError('Model archive exceeds extraction limits')
            if p.suffix not in ('.json','.pth'):continue
            dest = target/p
            if not dest.resolve().is_relative_to(target):raise ValueError('Unsafe archive path')
            dest.parent.mkdir(parents=True,exist_ok=True)
            source = t.extractfile(member)
            if source is None:raise ValueError('Missing archive member data')
            with source,dest.open('wb') as out:shutil.copyfileobj(source,out)


def setup(home, *, runtime_python=None, source=None, assets=None):
    home = Path(home).expanduser().resolve()
    home.mkdir(parents=True,exist_ok=True)
    checkout = home/'source'
    if not checkout.exists():
        if source:
            _run(['git','clone','--no-hardlinks',Path(source).resolve(),checkout],stdout=sys.stderr,stderr=sys.stderr)
        else:
            _run(['git','init',checkout],stdout=sys.stderr,stderr=sys.stderr)
            _run(['git','-C',checkout,'remote','add','origin',SOURCE])
            _run(['git','-C',checkout,'fetch','--depth','1','origin',COMMIT],stdout=sys.stderr,stderr=sys.stderr)
        _run(['git','-C',checkout,'checkout','--detach',COMMIT],stdout=sys.stderr,stderr=sys.stderr)
    current = _run(['git','-C',checkout,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    if current != COMMIT:
        raise ValueError('Model source checkout has an unexpected commit; select a fresh --model-home')
    _run(['git','-C',checkout,'diff','--exit-code'],stdout=subprocess.DEVNULL)
    if runtime_python:
        # Resolving the executable symlink would lose the venv's site-packages.
        python = Path(os.path.abspath(Path(runtime_python).expanduser()))
        if not python.is_file():raise ValueError('Model --python does not exist')
    else:
        python = home/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not python.exists():venv.EnvBuilder(with_pip=True).create(home/'venv')
        version=json.loads(_run([python,'-c','import sys,json; print(json.dumps(list(sys.version_info[:2])))'],capture_output=True,text=True).stdout)
        if not (3,10)<=tuple(version)<=(3,12):
            raise ValueError('The pinned model runtime needs Python 3.10–3.12; use --python with a compatible model environment')
        _run([python,'-m','pip','install','torch==2.5.1','nnunetv2==2.4.2',
              'dynamic_network_architectures==0.3.1','-e',checkout],stdout=sys.stderr,stderr=sys.stderr)
    probe = _run([python,'-c','import sys,torch,numpy,nnunetv2,nibabel,SimpleITK; print(sys.version.split()[0])'],capture_output=True,text=True)
    archives = home/'archives';archives.mkdir(exist_ok=True)
    data = home/'original_data'/'nnUNet'/'results'
    for name,expected in ASSETS.items():
        archive = archives/name
        if not archive.exists():
            partial=archive.with_suffix(archive.suffix+'.part')
            if assets:shutil.copyfile(Path(assets)/name,partial)
            else:
                url = 'https://github.com/Albertzry/SpineSegPlus/releases/download/weights/'+name
                request = urllib.request.Request(url,headers={'User-Agent':'disc-morphology-cli/0.1'})
                with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as out:
                    shutil.copyfileobj(response,out)
            if sha256(partial)!=expected:raise ValueError('Downloaded model archive SHA-256 mismatch: '+name)
            partial.replace(archive)
        if sha256(archive)!=expected:raise ValueError('Model archive SHA-256 mismatch: '+name)
        print('Verified model archive: '+name,file=sys.stderr,flush=True)
        safe_extract(archive,data,name.removesuffix('.tar.gz'))
    worker = Path(__file__).with_name('model_worker.py')
    _run([python,worker,'convert','--home',home],stdout=sys.stderr,stderr=sys.stderr)
    config = {'source_repository':SOURCE,'source_commit':COMMIT,'source':str(checkout),
              'python':str(python),'python_version':probe.stdout.strip(),'data':str(home/'safe_data'),
              'archives':ASSETS,'checkpoint_conversion':'weights_only; tensor equality verified'}
    (home/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return config


def config_for(home, *, verify=True):
    home = Path(home).expanduser().resolve()
    path = home/'config.json'
    if not path.is_file():
        raise ValueError(f'Models are not configured. Run: disc-morphology setup --model-home "{home}"')
    c = json.loads(path.read_text())
    if c.get('source_commit')!=COMMIT or not Path(c['python']).is_file():
        raise ValueError('Invalid model configuration; rerun setup')
    if verify:
        head = _run(['git','-C',c['source'],'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
        if head!=COMMIT:raise ValueError('Model source commit changed')
        _run(['git','-C',c['source'],'diff','--exit-code'],stdout=subprocess.DEVNULL)
        inventory = json.loads((home/'checkpoint-inventory.json').read_text())
        for item in inventory:
            if sha256(home/item['relative_path'])!=item['sha256']:
                raise ValueError('Model checkpoint changed: '+item['relative_path'])
    return c


def infer(input_dir, output, home, *, device='cpu', threads=2, timeout=3600):
    home = Path(home).expanduser().resolve()
    c = config_for(home)
    output = Path(output).resolve()
    if output.exists():raise ValueError('Model output already exists; choose a fresh run directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    worker = Path(__file__).with_name('model_worker.py')
    command = [c['python'],str(worker),'run','--home',str(home),'--input',str(Path(input_dir).resolve()),
               '--output',str(output),'--device',device,'--threads',str(threads)]
    env = dict(os.environ)
    env['PYTHONPATH'] = c['source']+os.pathsep+env.get('PYTHONPATH','')
    log_path=output.parent/'inference.log'
    started=time.monotonic()
    print('Running TotalSpineSeg → SpineSegPlus A/B; log: '+str(log_path),file=sys.stderr,flush=True)
    with log_path.open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt')
        try:
            code=process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait()
            raise
        if code:
            raise ValueError('Model inference failed; see '+str(log_path))
    candidate = output/'ldh_output'/'study.nii.gz'
    anatomy = output/'step2_output'/'study.nii.gz'
    if not candidate.is_file() or not anatomy.is_file():
        raise ValueError('Model returned without the expected mask/anatomy files')
    complete=json.loads((output/'completed.json').read_text())
    if (complete.get('model_pipeline_completed') is not True or complete.get('source_commit')!=COMMIT
            or complete.get('native_mask_sha256')!=sha256(candidate) or complete.get('anatomy_sha256')!=sha256(anatomy)):
        raise ValueError('Model completion marker/output integrity mismatch')
    return candidate,anatomy,{'source':'SpineSegPlus','source_commit':COMMIT,'inference_performed':True,
                             'device':device,'threads':threads,'model_home':str(home),
                             'elapsed_seconds':round(time.monotonic()-started,3),
                             'stage_a':json.loads((output/'ldh_stagea'/'study.json').read_text())}
