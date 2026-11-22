"""Known synthetic masks, never presented as a patient's MRI."""
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

from .geometry import measure_plane,classify_planes
from .pipeline import canonical,png,render_report
from .reference import _polygons


def payload():
    yy,xx=np.mgrid[:256,:256]
    images={};cases=[]
    for kind in ('protrusion','extrusion'):
        planes=[];records=[]
        for plane in ('sagittal','axial'):
            frames=[]
            for i,factor in enumerate((.45,.7,.9,1,.9,.7,.45)):
                if kind=='protrusion':
                    half=np.where(yy<=147,25-(yy-145)*3,19-(yy-147)*.4)*factor
                    mask=(yy>=143)&(yy<=178)&(abs(xx-128)<=half)
                else:
                    neck=(abs(xx-128)<=8*factor)&(yy>=143)&(yy<=163)
                    dome=((xx-128)/(22*factor))**2+((yy-170)/20)**2<=1
                    mask=neck|dome
                if plane=='sagittal':mask=mask.T
                baseline=[[145,65],[145,191]] if plane=='sagittal' else [[65,145],[191,145]]
                outside=[200,128] if plane=='sagittal' else [128,200]
                geometry={'rows':256,'columns':256,'spacing':[.5,.5],
                          'position':[0,0,0],'orientation':[1,0,0,0,1,0]}
                record=measure_plane(geometry,mask.tolist(),baseline,outside,plane=plane)
                ident=f'{kind}-{plane}-{i}'
                record['image_ref']={'image_id':ident,'series_id':'synthetic-'+plane,'frame_index':0}
                image=Image.new('L',(256,256),15);d=ImageDraw.Draw(image)
                if plane=='sagittal':d.rounded_rectangle((62,57,145,199),radius=20,fill=65,outline=115,width=2)
                else:d.ellipse((57,63,199,159),fill=65,outline=115,width=2)
                pixels=np.array(image);pixels[mask]=115
                images[ident]={'id':ident,'png':png(pixels),'width':256,'height':256,'spacing_mm':[.5,.5],
                               'label':str(i+1),'plane':plane,'geometry':geometry,'synthetic':True}
                xy=np.argwhere(mask).mean(0)[::-1].tolist()
                frames.append({'image_id':ident,'mask_polygons_xy':_polygons(mask),
                               'reference_line_xy':baseline,'candidate_mask_pixels':int(mask.sum()),
                               'focus_xy':xy,'record':record,'projection_allowed':True,'mask_source':'synthetic_known_geometry'})
                if i==3:records.append(record)
            planes.append({'plane':plane,'frames':frames,'selected_index':3})
        decision=classify_planes(records,source_is_synthetic=True)
        cases.append({'id':kind,'level':'Протрузия' if kind=='protrusion' else 'Экструзия',
                      'candidate_voxels':None,'planes':planes,'decision':decision,'reference':{'method':'known_synthetic_line'},
                      'reference_error':None,'status':'synthetic','clinical_measurements':{'B':None,'D':None,'H':None,'D_over_B':None},
                      'limitations':['Учебная геометрия с известным масштабом. Это не МРТ пациента и не нейросетевое предсказание.']})
    return {'schema_version':'1.0','tool_version':'0.1.0','study_id':'synthetic-demo',
            'generated_at':datetime.now(timezone.utc).isoformat(),'synthetic':True,
            'images':images,'cases':cases,'default_case_index':0,
            'model':{'source':'synthetic_known_geometry','inference_performed':False},
            'registration':{'axial_projection_performed':False,'registration_reviewed':False},
            'classification_scope':'synthetic_geometry','clinically_validated':False}


def build_demo(directory):
    out=Path(directory).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    data=payload()
    (out/'result.json').write_text(canonical(data))
    render_report(data,out/'report.html')
    return out/'report.html'
