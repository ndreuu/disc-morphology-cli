"""Optional-model subprocess. Safe checkpoint conversion; no training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback


def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def convert(home):
    import numpy as np
    import torch
    def legacy_scalar(*args):return np.core.multiarray.scalar(*args)
    legacy_scalar.__module__='numpy.core.multiarray';legacy_scalar.__name__='scalar'
    def modern_scalar(*args):return np.core.multiarray.scalar(*args)
    modern_scalar.__module__='numpy._core.multiarray';modern_scalar.__name__='scalar'
    allowed=[np.core.multiarray.scalar,legacy_scalar,modern_scalar,np.dtype,
             type(np.dtype(np.float32)),type(np.dtype(np.float64))]
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,tuple):return tuple(clean(v) for v in x)
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.dtype):return x.str
        if isinstance(x,torch.device):return str(x)
        if x is None or type(x) in (str,int,float,bool) or isinstance(x,torch.Tensor):return x
        raise ValueError('Unexpected checkpoint metadata type: '+str(type(x)))
    original_root=home/'original_data'
    inventory=[]
    for p in sorted(original_root.rglob('*.pth')):
        with torch.serialization.safe_globals(allowed):
            original=torch.load(p,map_location='cpu',weights_only=True)
        key='network_weights' if 'network_weights' in original else 'model' if 'model' in original else None
        if key is None:raise ValueError('Unknown checkpoint structure')
        keys=['network_weights','init_args','trainer_name','inference_allowed_mirroring_axes'] if key=='network_weights' else ['model']
        result=clean({k:original[k] for k in keys})
        dest=home/'safe_data'/p.relative_to(original_root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        torch.save(result,dest)
        loaded=torch.load(dest,map_location='cpu',weights_only=True)
        if set(original[key])!=set(loaded[key]):raise ValueError('Checkpoint tensor keys changed')
        for name,a in original[key].items():
            b=loaded[key][name]
            if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
                raise ValueError('Checkpoint tensor changed: '+name)
        inventory.append({'relative_path':str(dest.relative_to(home)),'sha256':file_hash(dest),
                          'original_sha256':file_hash(p),'tensor_equality_verified':True})
        print('Prepared inference weights: '+str(p.relative_to(original_root)),flush=True)
        del original,result,loaded
        gc.collect()
    if len(inventory)<4:raise ValueError('Expected all four model stages')
    for p in original_root.rglob('*.json'):
        if p.name not in ('dataset.json','plans.json'):continue
        dest=home/'safe_data'/p.relative_to(original_root)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        inventory.append({'relative_path':str(dest.relative_to(home)),'sha256':file_hash(dest)})
    (home/'checkpoint-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')


def run(args):
    home=args.home.resolve()
    config=json.loads((home/'config.json').read_text())
    threads=max(1,min(16,args.threads))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
        os.environ[key]=str(threads)
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
    os.environ['TOTALSPINESEG_DATA']=config['data']
    os.environ['nnUNet_results']=str(Path(config['data'])/'nnUNet'/'results')
    os.environ['nnUNet_raw']=str(args.output.parent/'nnunet_raw')
    os.environ['nnUNet_preprocessed']=str(args.output.parent/'nnunet_preprocessed')
    os.environ['MPLCONFIGDIR']=str(args.output.parent/'matplotlib')
    import torch
    original_set=torch.set_num_threads
    def capped(n):return original_set(min(threads,max(1,int(n))))
    torch.set_num_threads=capped;torch.set_num_threads(threads)
    sys.path.insert(0,config['source'])
    script=Path(config['source'])/'scripts'/'infer_ldh.py'
    sys.argv=[str(script),'--input-dir',str(args.input),'--output-dir',str(args.output),
              '--data-dir',config['data'],'--device',args.device,'--no-init','--max-workers','1',
              '--max-workers-nnunet','1','--disc-labels','92','93','94','95','100','--seed','42','--topk-b','1']
    runpy.run_path(str(script),run_name='__main__')
    required=[args.output/'ldh_output'/'study.nii.gz',args.output/'step2_output'/'study.nii.gz',
              args.output/'ldh_summary.json',args.output/'ldh_stagea'/'study.json']
    if not all(p.is_file() for p in required):
        raise ValueError('Model pipeline returned without all completed outputs')
    json.loads(required[2].read_text())
    complete={'model_pipeline_completed':True,'source_commit':config['source_commit'],
              'native_mask_sha256':file_hash(required[0]),'anatomy_sha256':file_hash(required[1])}
    (args.output/'completed.json').write_text(json.dumps(complete,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['convert','run']);ap.add_argument('--home',type=Path,required=True)
    ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda']);ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    if args.mode=='convert':convert(args.home.resolve())
    else:
        # Upstream multiprocessing objects can block CPython finalization on
        # macOS after the complete pipeline has returned (waitpid in __del__).
        # This is a disposable worker: outputs are closed and SHA-bound above.
        # Exit directly after flushing, rather than entering that finalizer.
        try:run(args)
        except BaseException:
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':main()
