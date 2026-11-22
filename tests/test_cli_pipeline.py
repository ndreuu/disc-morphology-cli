"""Synthetic DICOM only: selection, calibration, abstention and offline output."""
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from disc_morphology.cli import main,open_report
from disc_morphology.dicom import scan,select_series,load_series,sagittal_volume
from disc_morphology.mask_io import load_nifti,write_nifti,project_to_frame
from disc_morphology.pipeline import analyze,build_payload,render_report,verify
from disc_morphology.demo import payload as demo_payload


def dicom(path,*,series,position,orientation,description='T2',modality='MR',reference='1.2.3.99'):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=MRImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
    d=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    d.SOPClassUID=MRImageStorage;d.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    d.StudyInstanceUID='1.2.3.1';d.SeriesInstanceUID=series;d.FrameOfReferenceUID=reference
    d.SeriesDescription=description
    if modality:d.Modality=modality
    d.ImagePositionPatient=position;d.ImageOrientationPatient=orientation;d.PixelSpacing=[1,1]
    d.Rows=64;d.Columns=64;d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME2'
    d.BitsAllocated=16;d.BitsStored=16;d.HighBit=15;d.PixelRepresentation=0
    d.PixelData=np.arange(64*64,dtype='<u2').reshape(64,64).tobytes()
    d.save_as(path,enforce_file_format=True)


def fixture(root):
    source=root/'dicom';source.mkdir()
    for i in range(7):
        dicom(source/f'sag-{i}.dcm',series='1.2.3.2',position=[-6+i*2,-25,40],orientation=[0,1,0,0,0,-1])
    for i in range(3):
        dicom(source/f'ax-{i}.dcm',series='1.2.3.3',position=[-32,-25,10-i*2],orientation=[1,0,0,0,1,0],reference='1.2.3.98')
    groups,_=scan(source);sag=load_series(select_series(groups,'sagittal'))
    vol,aff=sagittal_volume(sag)
    anatomy=np.zeros(vol.shape,dtype='uint8')
    anatomy[8:29,10:43,:]=43;anatomy[35:55,10:43,:]=44;anatomy[29:35,15:40,:]=94
    mask=np.zeros(vol.shape,dtype='uint8')
    mask[29:36,40:46,2:5]=1;mask[24:41,45:51,2:5]=1
    cp=root/'candidate.nii.gz';ap=root/'anatomy.nii.gz'
    write_nifti(cp,mask,aff);write_nifti(ap,anatomy,aff)
    return source,cp,ap


class ScriptData(HTMLParser):
    def __init__(self):super().__init__();self.collect=False;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='script' and dict(attrs).get('id')=='report-data':self.collect=True
    def handle_endtag(self,tag):
        if tag=='script':self.collect=False
    def handle_data(self,data):
        if self.collect:self.parts.append(data)


class InputTests(unittest.TestCase):
    def test_auto_t2_selection_uses_headers_and_distinguishes_t1(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);source,_,_=fixture(r)
            dicom(source/'t1.dcm',series='1.2.3.4',position=[0,-25,40],orientation=[0,1,0,0,0,-1],description='T1')
            groups,ignored=scan(source)
            self.assertEqual(select_series(groups,'sagittal')['uid'],'1.2.3.2')
            self.assertEqual(select_series(groups,'axial')['uid'],'1.2.3.3')
            self.assertEqual(ignored,0)

    def test_ambiguous_t2_requires_explicit_selection(self):
        groups={'a':{'uid':'a','plane':'sagittal','description':'T2','echo_time':0},
                'b':{'uid':'b','plane':'sagittal','description':'T2','echo_time':0}}
        with self.assertRaisesRegex(ValueError,'Cannot select'):select_series(groups,'sagittal')
        self.assertEqual(select_series(groups,'sagittal','b')['uid'],'b')

    def test_missing_modality_requires_user_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dicom(root/'one.dcm',series='1.2.3.2',position=[0,0,40],orientation=[0,1,0,0,0,-1],modality='')
            groups,_=scan(root);s=select_series(groups,'sagittal')
            with self.assertRaisesRegex(ValueError,'allow-missing-modality'):load_series(s)
            self.assertEqual(load_series(s,allow_missing_modality=True)[0]['modality_status'],'user_declared_mr')

    def test_flipped_sagittal_preserves_physical_pixel_correspondence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dicom(root/'flip.dcm',series='1.2.3.2',position=[0,38,40],orientation=[0,-1,0,0,0,-1])
            groups,_=scan(root);f=load_series(select_series(groups,'sagittal'))[0]
            self.assertEqual(f['display_transform'],['flip_columns'])
            np.testing.assert_allclose(f['position'],[0,-25,40])
            self.assertEqual(f['raw_pixels'][0,0],63)
            self.assertEqual(f['orientation'],[0.,1.,0.,0.,0.,-1.])

    def test_irregular_slices_are_rejected_instead_of_inventing_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,x in enumerate((0,2,7)):
                dicom(root/f'{i}.dcm',series='1.2.3.2',position=[x,-25,40],orientation=[0,1,0,0,0,-1])
            groups,_=scan(root);frames=load_series(select_series(groups,'sagittal'))
            with self.assertRaisesRegex(ValueError,'Irregular'):sagittal_volume(frames)

    def test_native_volume_writer_roundtrip_preserves_pixels_and_affine(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);source,_,_=fixture(r)
            groups,_=scan(source);sag=load_series(select_series(groups,'sagittal'))
            volume,aff=sagittal_volume(sag);p=r/'source.nii.gz';write_nifti(p,volume,aff)
            arr,a=load_nifti(p)
            np.testing.assert_array_equal(arr,volume);np.testing.assert_allclose(a,aff)
            np.testing.assert_array_equal(arr[:,:,3],sag[3]['raw_pixels'])

    def test_rectangular_stack_preserves_audited_rows_columns_normal_order(self):
        frames=[]
        for i,x in enumerate((-2,0,2)):
            frames.append({'rows':20,'columns':30,'spacing':[2.,3.],
                           'orientation':[0,1,0,0,0,-1],'position':[x,-25,40],
                           'raw_pixels':np.arange(600,dtype=np.float32).reshape(20,30)+i*1000})
        volume,affine=sagittal_volume(frames)
        self.assertEqual(volume.shape,(20,30,3))
        np.testing.assert_array_equal(volume[:,:,0],frames[-1]['raw_pixels'])
        for f in frames:np.testing.assert_array_equal(project_to_frame(volume,affine,f),f['raw_pixels'])

    def test_screenshots_do_not_get_a_fake_physical_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'screenshot.png').write_bytes(b'not a dicom')
            with self.assertRaisesRegex(ValueError,'No DICOM'):scan(tmp)


class PipelineTests(unittest.TestCase):
    def test_external_mask_native_measurements_and_axial_projection_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,cp,ap=fixture(root)
            report,data=analyze(source,root/'run',model_home=root/'no-models',candidate_path=cp,anatomy_path=ap,
                                allow_unverified_registration=True)
            self.assertTrue(report.is_file());self.assertFalse(data['model']['inference_performed'])
            c=next(c for c in data['cases'] if c['level']=='L3-L4')
            sag=c['planes'][0];r=sag['frames'][sag['selected_index']]['record']
            self.assertTrue(r['geometry_measurable']);self.assertGreater(r['candidate_displaced_width_mm'],r['candidate_base_width_mm'])
            self.assertEqual(c['decision']['geometric_candidate'],'extrusion')
            self.assertEqual(c['decision']['clinical_classification'],'indeterminate')
            ax=c['planes'][1]
            self.assertTrue(any(f['candidate_mask_pixels'] for f in ax['frames']))
            self.assertTrue(all(not f['record']['geometry_measurable'] for f in ax['frames']))
            self.assertTrue(verify(report.parent)['ok'])
            report.write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Integrity'):verify(report.parent)

    def test_different_frames_of_reference_do_not_silently_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,cp,ap=fixture(root)
            _,data=analyze(source,root/'run',model_home=root/'unused',candidate_path=cp,anatomy_path=ap)
            self.assertFalse(data['registration']['axial_projection_performed'])
            self.assertTrue(all(not f['mask_polygons_xy'] for c in data['cases'] for p in c['planes'] if p['plane']=='axial' for f in p['frames']))

    def test_empty_model_mask_is_not_a_no_herniation_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,cp,ap=fixture(root)
            a,aff=load_nifti(cp);write_nifti(cp,np.zeros(a.shape,dtype='uint8'),aff)
            _,data=analyze(source,root/'run',model_home=root/'unused',candidate_path=cp,anatomy_path=ap)
            self.assertTrue(all(c['decision']['clinical_classification']=='indeterminate' and not c['decision']['absence_assessed'] for c in data['cases']))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,cp,ap=fixture(root);out=root/'run';out.mkdir();marker=out/'keep';marker.write_text('unchanged')
            with self.assertRaisesRegex(ValueError,'already exists'):
                analyze(source,out,model_home=root/'unused',candidate_path=cp,anatomy_path=ap)
            self.assertEqual(marker.read_text(),'unchanged')

    def test_html_embeds_exact_json_without_a_script_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=demo_payload();data['study_id']='</script><img src=x onerror=alert(1)>'
            p=Path(tmp)/'report.html';render_report(data,p);h=p.read_text()
            parser=ScriptData();parser.feed(h);decoded=json.loads(''.join(parser.parts))
            self.assertEqual(decoded['study_id'],data['study_id'])
            self.assertNotIn('<img src=x',h)
            self.assertNotIn('<script src=',h)
            self.assertNotIn('<link rel="stylesheet"',h)

    def test_cli_no_open_and_browser_open_use_the_completed_report(self):
        with tempfile.TemporaryDirectory() as tmp,patch('disc_morphology.cli.webbrowser.open',return_value=True) as browser:
            out=Path(tmp)/'demo'
            with redirect_stdout(io.StringIO()):self.assertEqual(main(['demo','-o',str(out),'--no-open']),0)
            browser.assert_not_called();open_report(out/'report.html',False)
            browser.assert_called_once_with((out/'report.html').resolve().as_uri())

    def test_cli_setup_is_not_claimed_successful_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_,_=fixture(root)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
                result=main(['analyze',str(source),'-o',str(root/'run'),'--model-home',str(root/'missing'),'--no-open'])
            self.assertEqual(result,2);self.assertIn('not configured',errors.getvalue())
            self.assertFalse((root/'run'/'report.html').exists())


if __name__=='__main__':unittest.main()
