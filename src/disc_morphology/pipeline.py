"""General DICOM → model mask → per-frame research geometry → report."""
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from . import __version__
from .dicom import load_series, scan, select_series, series_info, sagittal_volume, sha256
from .geometry import classify_planes, measure_plane
from .mask_io import load_nifti, project_to_frame, write_nifti
from .reference import _reference, _line, _polygons

LEVELS={92:'L1-L2',93:'L2-L3',94:'L3-L4',95:'L4-L5',100:'L5-S1'}


def canonical(data):
    return json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2,sort_keys=True)+'\n'


def png(pixels):
    f=BytesIO();Image.fromarray(pixels.astype('uint8')).save(f,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode('ascii')


def image_payload(frame,index,plane):
    return {'id':frame['image_ref']['image_id'],'png':png(frame['pixels']),
            'width':frame['columns'],'height':frame['rows'],'spacing_mm':frame['spacing'],
            'label':f'{index+1}','plane':plane,'sha256':frame['sha256'],
            'geometry':{k:frame[k] for k in ('position','orientation','spacing','rows','columns')},
            'display_transform':frame.get('display_transform',[])}


def split_by_level(candidate,affine,anatomy,aa):
    """Assign binary candidate voxels to the nearest automatic disc centroid.

    Assignment does not establish a migrated fragment's disc of origin.
    """
    centers={}
    for label in LEVELS:
        vox=np.argwhere(anatomy==label)
        if len(vox):centers[label]=(aa@np.append(np.median(vox,axis=0),1))[:3]
    assigned=np.zeros(candidate.shape,dtype=np.uint8)
    vox=np.argwhere(candidate>0)
    if len(vox) and not centers:
        raise ValueError('Candidate present but lumbar disc anatomy could not be identified')
    if len(vox):
        labels=list(centers)
        for start in range(0,len(vox),4096):
            chunk=vox[start:start+4096]
            world=chunk@affine[:3,:3].T+affine[:3,3]
            distances=np.stack([np.linalg.norm(world-centers[k],axis=1) for k in labels],axis=1)
            choices=np.asarray(labels,dtype=np.uint8)[distances.argmin(axis=1)]
            assigned[tuple(chunk.T)]=choices
    return assigned


def frame_payload(frame,mask,reference,plane,*,source_uncertainty,projection_allowed=True):
    refline=None;record=None
    if reference is not None and (plane!='axial' or projection_allowed):
        try:
            refline,outside=_line(frame,reference)
            u=max(math.hypot(*frame['spacing']),reference['fit_rms_mm'],source_uncertainty if plane=='axial' else 0)
            record=measure_plane(frame,mask.tolist(),refline,outside,plane=plane,uncertainty_mm=u)
        except ValueError as e:
            record={'geometry_measurable':False,'geometric_candidate':'indeterminate','quality_flags':['reference_not_available_on_frame'],
                    'detail':str(e),'plane':plane}
    if record is None:
        record={'geometry_measurable':False,'geometric_candidate':'indeterminate',
                'quality_flags':['registration_unverified' if plane=='axial' and not projection_allowed else 'reference_unavailable'],'plane':plane}
    record['image_ref']=frame['image_ref']
    record['quality_flags']=list(record.get('quality_flags',[]))+['reference_unreviewed','segmentation_needs_review']
    if not np.any(mask) and 'empty_mask' not in record['quality_flags']:
        record['quality_flags'].append('empty_mask')
    if plane=='axial':
        record['quality_flags'].append('projected_sagittal_mask_not_independent_axial_segmentation')
        record['geometry_measurable']=False
        record['geometric_candidate']='indeterminate'
        record['status']='indeterminate'
    coordinates=np.argwhere(mask)
    center=coordinates.mean(0)[::-1].tolist() if len(coordinates) else [frame['columns']/2,frame['rows']/2]
    return {'image_id':frame['image_ref']['image_id'],'mask_polygons_xy':_polygons(mask),
            'candidate_mask_pixels':int(mask.sum()),'focus_xy':center,'reference_line_xy':refline,
            'record':record,'projection_allowed':projection_allowed,
            'mask_source':'sagittal_model_mask' if plane=='sagittal' else 'sagittal_mask_projected' if projection_allowed else 'no_projection'},record


def build_payload(sag,ax,candidate,ca,anatomy,aa,*,model_info=None,allow_unverified_registration=False,study_id='local-study'):
    assigned=split_by_level(candidate,ca,anatomy,aa)
    images={}
    for plane,frames in (('sagittal',sag),('axial',ax)):
        images.update({f['image_ref']['image_id']:image_payload(f,i,plane) for i,f in enumerate(frames)})
    srefs={f['frame_of_reference_uid'] for f in sag}
    arefs={f['frame_of_reference_uid'] for f in ax}
    same_ref=bool(ax and len(srefs)==1 and srefs==arefs and '' not in srefs)
    projection_allowed=same_ref or allow_unverified_registration
    _,sag_affine=sagittal_volume(sag)
    uncertainty=max(float(np.linalg.norm(ca[:3,:3],axis=0).max()),float(np.linalg.norm(sag_affine[:3,:3],axis=0).max()))
    cases=[]
    for label,level in LEVELS.items():
        level_mask=(assigned==label).astype('uint8')
        voxels=int(level_mask.sum())
        print(f'Geometry {level}: {voxels} candidate voxels',file=sys.stderr,flush=True)
        reference=None;ref_error=None
        if voxels:
            try:reference=_reference(sag,anatomy,aa,label)
            except ValueError as e:ref_error=str(e)
        else:ref_error='No candidate mask; absence of herniation was not assessed'
        planes=[];representatives=[]
        for plane,frames in (('sagittal',sag),('axial',ax)):
            payloads=[];records=[]
            for f in frames:
                mask=project_to_frame(level_mask,ca,f)>0 if voxels and (plane!='axial' or projection_allowed) else np.zeros((f['rows'],f['columns']),dtype=bool)
                p,r=frame_payload(f,mask,reference,plane,source_uncertainty=uncertainty,projection_allowed=projection_allowed)
                if model_info and not model_info.get('inference_performed'):
                    p['mask_source']='provided_3d_mask' if plane=='sagittal' else 'provided_mask_projected' if projection_allowed else 'no_projection'
                payloads.append(p);records.append(r)
            selected=max(range(len(payloads)),key=lambda i:payloads[i]['candidate_mask_pixels']) if payloads else 0
            if records:representatives.append(records[selected])
            planes.append({'plane':plane,'frames':payloads,'selected_index':selected})
        decision=classify_planes(representatives)
        decision['full_multislice_morphology_assessed']=False
        decision['representative_frame_selection']='maximum_candidate_mask_pixels_per_plane'
        cases.append({'id':str(label),'level':level,'candidate_voxels':voxels,'planes':planes,'decision':decision,
                      'reference':reference,'reference_error':ref_error,
                      'status':'candidate_needs_review' if voxels else 'no_candidate_mask_not_an_absence_diagnosis',
                      'clinical_measurements':{'B':None,'D':None,'H':None,'D_over_B':None},
                      'limitations':['Граница построена приближённо по соседним позвонкам; остеофиты и апофизарные кольца отдельно не определены.',
                                     'D — максимальная связная ширина параллельно B за первой полосой разрешения; полный поиск размера во всех направлениях не выполнен.',
                                     'Аксиальный контур — перенос сагиттальной маски, без независимой аксиальной сегментации.',
                                     'Нумерация и отнесение маски к ближайшему диску автоматически предложены и не подтверждены.',
                                     'Нет оценки непрерывности, миграции, секвестрации или корешковой компрессии.']+
                                    (['Обучающее покрытие опубликованных весов для верхних трёх поясничных уровней не подтверждено.'] if label in (92,93,94) else [])})
    default=max(range(len(cases)),key=lambda i:cases[i]['candidate_voxels'])
    return {'schema_version':'1.0','tool_version':__version__,'study_id':study_id,
            'generated_at':datetime.now(timezone.utc).isoformat(),'synthetic':False,
            'images':images,'cases':cases,'default_case_index':default,
            'model':{k:v for k,v in (model_info or {'source':'provided_mask','inference_performed':False}).items() if k!='model_home'},
            'registration':{'shared_frame_of_reference_uid':same_ref,'unverified_projection_requested':allow_unverified_registration,
                            'axial_projection_performed':bool(ax and projection_allowed),'registration_reviewed':False},
            'formulas':{'B':'length of the connected chord at the reference line',
                        'D':'maximum connected chord parallel to B, outside the first resolution band',
                        'H':'maximum perpendicular outward distance from the reference line',
                        'D_over_B':'D / B','extrusion':'D - B > 2u in at least one usable plane',
                        'protrusion':'B - D > 2u in both usable independent planes',
                        'indeterminate':'otherwise; clinical classification remains gated by unreviewed anatomy'},
            'classification_scope':'unvalidated_research_candidate_geometry','clinically_validated':False}


def render_report(payload,path):
    from importlib.resources import files
    assets=files('disc_morphology').joinpath('assets')
    html=assets.joinpath('report.html').read_text(encoding='utf-8')
    data=canonical(payload).replace('<','\\u003c')
    html=html.replace('/* REPORT_STYLES */',assets.joinpath('report.css').read_text(encoding='utf-8'))
    html=html.replace('/* REPORT_DATA */',data)
    html=html.replace('/* REPORT_SCRIPT */',assets.joinpath('report.js').read_text(encoding='utf-8'))
    Path(path).write_text(html,encoding='utf-8')


def analyze(input_dir,output,*,model_home,candidate_path=None,anatomy_path=None,sagittal_series=None,axial_series=None,
            allow_missing_modality=False,allow_unverified_registration=False,device='cpu',threads=2,timeout=3600):
    started=time.monotonic()
    root=Path(input_dir).expanduser().resolve();out=Path(output).expanduser().resolve()
    if out.exists():raise ValueError('Output directory already exists; choose a new directory')
    if out==root or out.is_relative_to(root):raise ValueError('Output must be outside the input DICOM directory')
    groups,ignored=scan(root)
    s=select_series(groups,'sagittal',sagittal_series)
    a=select_series(groups,'axial',axial_series,optional=True)
    if a and s['study_uid']!=a['study_uid']:raise ValueError('Selected series belong to different studies')
    if len({g['study_uid'] for g in groups.values()})>1 and not sagittal_series:
        raise ValueError('Input contains multiple studies; select a study directory or explicit series UIDs')
    sag=load_series(s,allow_missing_modality=allow_missing_modality)
    ax=load_series(a,allow_missing_modality=allow_missing_modality) if a else []
    print(f'Read {len(sag)} sagittal and {len(ax)} axial DICOM frames',file=sys.stderr,flush=True)
    volume,affine=sagittal_volume(sag)
    out.mkdir(parents=True)
    try:
        input_volume=out/'model_input'/'study.nii.gz'
        write_nifti(input_volume,volume,affine)
        reloaded,stored_affine=load_nifti(input_volume)
        if any(not np.array_equal(project_to_frame(reloaded,stored_affine,f),f['raw_pixels']) for f in sag):
            raise ValueError('NIfTI conversion did not preserve every native sagittal DICOM pixel in physical coordinates')
        if bool(candidate_path)!=bool(anatomy_path):raise ValueError('--candidate and --anatomy must be supplied together')
        if candidate_path:
            cp=Path(candidate_path).expanduser().resolve();ap=Path(anatomy_path).expanduser().resolve()
            info={'source':'provided_mask','inference_performed':False}
        else:
            from .backend import infer
            cp,ap,info=infer(input_volume.parent,out/'segmentation',model_home,device=device,threads=threads,timeout=timeout)
        candidate,ca=load_nifti(cp);anatomy,aa=load_nifti(ap)
        if not np.isin(candidate,[0,1]).all():raise ValueError('Candidate mask must be binary 0/1')
        if info['inference_performed'] and (candidate.shape!=volume.shape or not np.allclose(ca,affine,atol=.002,rtol=1e-5)):
            raise ValueError('Model output does not match its input DICOM volume geometry')
        info.update({'candidate_sha256':sha256(cp),'anatomy_sha256':sha256(ap)})
        sid='study-'+hashlib.sha256(s['study_uid'].encode()).hexdigest()[:12]
        payload=build_payload(sag,ax,candidate,ca,anatomy,aa,model_info=info,
                              allow_unverified_registration=allow_unverified_registration,study_id=sid)
        payload['input_summary']={'sagittal_frames':len(sag),'axial_frames':len(ax),'ignored_nonimage_files':ignored}
        payload['elapsed_seconds']=round(time.monotonic()-started,3)
        (out/'result.json').write_text(canonical(payload))
        render_report(payload,out/'report.html')
        sources=[{'path':f['path'],'sha256':f['sha256']} for f in sag+ax]
        manifest={'tool_version':__version__,'input':str(root),'sources':sources,
                  'selected_series':{'sagittal':s['uid'],'axial':a['uid'] if a else None},
                  'candidate':str(cp),'candidate_sha256':sha256(cp),'anatomy':str(ap),'anatomy_sha256':sha256(ap),
                  'model_input':str(input_volume),'model_input_sha256':sha256(input_volume),
                  'result_sha256':sha256(out/'result.json'),'report_sha256':sha256(out/'report.html')}
        (out/'run.json').write_text(canonical(manifest))
        return out/'report.html',payload
    except Exception as e:
        (out/'error.json').write_text(canonical({'status':'failed','error':str(e),'clinical_classification':'indeterminate'}))
        raise


def verify(directory):
    directory=Path(directory).expanduser().resolve()
    m=json.loads((directory/'run.json').read_text())
    checks=[(directory/'result.json',m['result_sha256']),(directory/'report.html',m['report_sha256']),
            (Path(m['candidate']),m['candidate_sha256']),(Path(m['anatomy']),m['anatomy_sha256'])]
    if m.get('model_input'):checks.append((Path(m['model_input']),m['model_input_sha256']))
    checks.extend((Path(i['path']),i['sha256']) for i in m['sources'])
    for path,expected in checks:
        if sha256(path)!=expected:raise ValueError('Integrity mismatch: '+str(path))
    return {'ok':True,'checked_files':len(checks),'scope':'file integrity only; no new inference or clinical validation'}
