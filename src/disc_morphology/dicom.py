"""Read conventional grayscale DICOM; never infer physical scale from a JPEG."""
import hashlib
import io
import math
from pathlib import Path
import re

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from ._spatial import _axes


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_pixels(pixels):
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError('Expected a finite grayscale image')
    low, high = np.percentile(pixels, [.1, 99.9])
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return np.rint(np.clip((pixels-low)/(high-low), 0, 1)*255).astype(np.uint8)


def scan(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Input must be a directory containing original DICOM files')
    groups, ignored = {}, 0
    for i, path in enumerate(sorted(root.rglob('*'))):
        if i > 100000:
            raise ValueError('Too many input entries; select one study directory')
        if not path.is_file() or path.is_symlink() or path.name.startswith('.'):
            continue
        try:
            d = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError):
            ignored += 1
            continue
        if not all(hasattr(d, x) for x in ('StudyInstanceUID','SeriesInstanceUID','Rows','Columns','ImageOrientationPatient','ImagePositionPatient','PixelSpacing')):
            ignored += 1
            continue
        orientation = np.asarray(d.ImageOrientationPatient, dtype=float)
        if orientation.shape != (6,) or not np.isfinite(orientation).all():
            raise ValueError('Invalid DICOM orientation')
        n = np.cross(orientation[:3], orientation[3:])
        plane = 'sagittal' if abs(n[0]) > .85 else 'axial' if abs(n[2]) > .65 else 'other'
        uid = str(d.SeriesInstanceUID)
        item = groups.setdefault(uid, {'uid':uid,'study_uid':str(d.StudyInstanceUID),'description':str(getattr(d,'SeriesDescription','')),
                                      'plane':plane,'echo_time':float(getattr(d,'EchoTime',0) or 0),'files':[]})
        if item['study_uid'] != str(d.StudyInstanceUID) or item['plane'] != plane:
            raise ValueError('Inconsistent geometry/study within a DICOM series')
        item['files'].append(path)
    if not groups:
        raise ValueError('No DICOM images with spatial geometry found; JPEG/PNG screenshots cannot supply calibrated measurements')
    return groups, ignored


def series_info(groups):
    return [{'uid':g['uid'],'study_uid':g['study_uid'],'plane':g['plane'],
             'description':g['description'],'echo_time':g['echo_time'],'frames':len(g['files'])}
            for g in groups.values()]


def select_series(groups, plane, explicit=None, *, optional=False):
    if explicit:
        if explicit not in groups or groups[explicit]['plane'] != plane:
            raise ValueError(f'--{plane}-series does not identify a {plane} series; run inspect')
        return groups[explicit]
    eligible = [g for g in groups.values() if g['plane'] == plane]
    t2 = [g for g in eligible if re.search(r'(?i)(?:^|[^a-z])t2(?:w|[^a-z]|$)',g['description']) or g['echo_time'] >= 60]
    if len(t2) == 1:
        return t2[0]
    if optional and not eligible:
        return None
    raise ValueError(f'Cannot select one {plane} T2 series automatically; run inspect and provide --{plane}-series UID')


def load_series(series, *, allow_missing_modality=False):
    frames = []
    for path in series['files']:
        if path.stat().st_size > 512*1024*1024:
            raise ValueError('DICOM file exceeds the supported size')
        data = path.read_bytes()
        d = pydicom.dcmread(io.BytesIO(data))
        modality = str(getattr(d,'Modality',''))
        if modality != 'MR' and not (not modality and allow_missing_modality):
            raise ValueError('Expected MR DICOM; anonymized MR without Modality requires --allow-missing-modality')
        if (int(getattr(d,'NumberOfFrames',1)) != 1 or int(getattr(d,'SamplesPerPixel',0)) != 1
                or str(getattr(d,'PhotometricInterpretation','')) not in ('MONOCHROME1','MONOCHROME2')
                or hasattr(d,'PerFrameFunctionalGroupsSequence')):
            raise ValueError('Only conventional single-frame grayscale MR is supported; enhanced/multiframe DICOM needs explicit conversion')
        rows, columns = int(d.Rows), int(d.Columns)
        if min(rows,columns) < 2 or rows*columns > 16*1024*1024:
            raise ValueError('Unsupported DICOM matrix size')
        raw = np.asarray(d.pixel_array,dtype=np.float32)
        if raw.shape != (rows,columns) or not np.isfinite(raw).all():
            raise ValueError('DICOM pixels disagree with declared geometry')
        raw = raw*float(getattr(d,'RescaleSlope',1))+float(getattr(d,'RescaleIntercept',0))
        f = {'rows':rows,'columns':columns,'position':[float(v) for v in d.ImagePositionPatient],
             'orientation':[float(v) for v in d.ImageOrientationPatient], 'spacing':[float(v) for v in d.PixelSpacing],
             'frame_of_reference_uid':str(getattr(d,'FrameOfReferenceUID','')),
             'sha256':hashlib.sha256(data).hexdigest(),'path':str(path),'raw_pixels':raw,
             'modality_status':'dicom_declared' if modality else 'user_declared_mr','display_transform':[]}
        _axes(f)
        if series['plane'] == 'sagittal':
            # Preserve exact pixels and update IPP/IOP when changing display axes.
            c,r = np.array(f['orientation'][:3]),np.array(f['orientation'][3:])
            if abs(c[2]) > .8 and abs(r[1]) > .8:
                f['raw_pixels'] = f['raw_pixels'].T.copy()
                f['rows'],f['columns'] = f['columns'],f['rows']
                f['spacing'] = f['spacing'][::-1]
                f['orientation'] = r.tolist()+c.tolist()
                f['display_transform'].append('transpose')
            for axis, flip in ((1,f['orientation'][1]<0),(0,f['orientation'][5]>0)):
                if not flip: continue
                vector = np.array(f['orientation'][:3] if axis==1 else f['orientation'][3:])
                extent = (f['columns']-1)*f['spacing'][1] if axis==1 else (f['rows']-1)*f['spacing'][0]
                f['position'] = (np.array(f['position'])+vector*extent).tolist()
                if axis==1:f['orientation'] = (-vector).tolist()+f['orientation'][3:]
                else:f['orientation'] = f['orientation'][:3]+(-vector).tolist()
                f['raw_pixels'] = np.flip(f['raw_pixels'],axis=axis).copy()
                f['display_transform'].append('flip_columns' if axis==1 else 'flip_rows')
            if f['orientation'][1] < .8 or f['orientation'][5] > -.8:
                raise ValueError('Sagittal orientation is too oblique for the posterior-reference estimator')
        f['pixels'] = normalize_pixels(f['raw_pixels'])
        if str(d.PhotometricInterpretation)=='MONOCHROME1':f['pixels']=255-f['pixels']
        f['image_ref'] = {'series_id':series['uid'],'image_id':f['sha256'][:16],'frame_index':0}
        frames.append(f)
    key = (lambda f:f['position'][0]) if series['plane']=='sagittal' else (lambda f:-f['position'][2])
    frames.sort(key=key)
    if len({f['sha256'] for f in frames})!=len(frames):
        raise ValueError('Duplicate DICOM frames in the selected series')
    return frames


def sagittal_volume(frames):
    if len(frames) < 3:
        raise ValueError('At least three spatially distinct sagittal slices are required')
    # Preserve the already audited model preprocessing: rows, columns, slices,
    # with slices ordered along the DICOM normal. Viewer order is independent.
    normal=np.asarray(_axes(frames[0])[3])
    ordered=sorted(frames,key=lambda f:float(np.asarray(f['position'])@normal))
    first = ordered[0]
    p,c,r,n = _axes(first)
    p,c,r,n = map(np.asarray,(p,c,r,n))
    positions = []
    for f in ordered:
        if (f['rows']!=first['rows'] or f['columns']!=first['columns']
                or not np.allclose(f['orientation'],first['orientation'],atol=1e-5,rtol=0)
                or not np.allclose(f['spacing'],first['spacing'],atol=1e-5,rtol=0)):
            raise ValueError('Sagittal geometry is inconsistent; explicit resampling is required')
        delta = np.asarray(f['position'])-p
        projection = float(delta@n)
        if np.linalg.norm(delta-projection*n) > .05:
            raise ValueError('Sagittal stack has in-plane drift; explicit resampling is required')
        positions.append(projection)
    gaps = np.diff(positions)
    step = float(np.median(gaps))
    if abs(step) < .01 or not np.allclose(gaps,step,atol=.02,rtol=.005):
        raise ValueError('Irregular/duplicate sagittal slice positions; explicit resampling is required')
    steps=np.diff(np.asarray([f['position'] for f in ordered]),axis=0)
    if not np.allclose(steps,steps[0],atol=.001,rtol=.0001):
        raise ValueError('Irregular sagittal interslice vectors; explicit resampling is required')
    affine = np.eye(4)
    affine[:3,:3] = np.column_stack((r,c,steps[0]))
    affine[:3,3] = p
    affine = np.diag([-1,-1,1,1])@affine
    volume = np.stack([f['raw_pixels'] for f in ordered],axis=2).astype(np.float32)
    return volume,affine
