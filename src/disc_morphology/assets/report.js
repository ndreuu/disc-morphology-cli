(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('report-data').textContent);
  const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
  const state={caseIndex:data.default_case_index||0,indices:{},plane:'sagittal',zoom:true,radius:24,mask:true,reference:true,measures:true};
  const cache=new Map(),drawTokens={sagittal:0,axial:0};
  const names={protrusion:'Протрузия',extrusion:'Экструзия',indeterminate:'Не определено'};
  const fmt=(value,digits=2)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('ru-RU',{minimumFractionDigits:digits,maximumFractionDigits:digits}):'—';
  const text=(selector,value)=>{$(selector).textContent=value;};
  const current=()=>data.cases[state.caseIndex];
  const planeData=p=>current().planes.find(x=>x.plane===p)||{frames:[],selected_index:0};
  const frame=p=>planeData(p).frames[state.indices[p]||0];
  const image=p=>data.images[frame(p)?.image_id];
  const panel=p=>$(`.image-panel[data-plane="${p}"]`);

  function loadImage(src){
    if(!cache.has(src))cache.set(src,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=src;}));
    return cache.get(src);
  }

  function selectCase(index){
    state.caseIndex=index;state.plane='sagittal';
    for(const p of ['sagittal','axial']){
      const entries=planeData(p);let selected=entries.selected_index||0;
      const numeric=f=>['candidate_base_width_mm','candidate_displaced_width_mm','candidate_depth_mm'].every(k=>typeof f?.record?.[k]==='number'&&Number.isFinite(f.record[k]));
      // Prefer a frame with visible estimates. This only chooses the initial
      // view; stored representative decisions and all quality gates stay intact.
      if(!numeric(entries.frames[selected])){
        const available=entries.frames.map((f,i)=>({f,i})).filter(x=>numeric(x.f));
        if(available.length)selected=available.reduce((a,b)=>a.f.candidate_mask_pixels>=b.f.candidate_mask_pixels?a:b).i;
      }
      state.indices[p]=selected;
    }
    $$('#levels button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    const list=$('#limitations');list.replaceChildren();
    for(const message of current().limitations||[]){const li=document.createElement('li');li.textContent=message;list.append(li);}
    document.title=current().level+' · DISC Morphology';
    render();
  }

  function setFrame(plane,index){
    const frames=planeData(plane).frames;
    if(!frames.length)return;
    state.indices[plane]=Math.max(0,Math.min(frames.length-1,index));state.plane=plane;
    render();
  }

  function render(){
    for(const p of ['sagittal','axial']){
      const node=panel(p),entries=planeData(p).frames,f=frame(p),im=image(p);
      node.classList.toggle('active',state.plane===p);
      const range=node.querySelector('.slice');range.max=Math.max(0,entries.length-1);range.value=state.indices[p]||0;range.disabled=!entries.length;
      node.querySelector('.slice-count').textContent=entries.length?`${(state.indices[p]||0)+1} / ${entries.length}`:'Нет серии';
      node.querySelector('[data-step="-1"]').disabled=!entries.length||(state.indices[p]||0)===0;
      node.querySelector('[data-step="1"]').disabled=!entries.length||(state.indices[p]||0)>=entries.length-1;
      node.querySelector('.image-type').textContent=data.synthetic?'Схема':p==='sagittal'?'Маска кандидата':f?.projection_allowed?'Перенос маски':'Исходный DICOM';
      let note=!f?'Аксиальная серия не предоставлена.':data.synthetic?'Синтетическая геометрия · не МРТ пациента':p==='sagittal'?'Автоматический контур; границы требуют проверки.':f.projection_allowed?'Проекция сагиттальной маски. Независимой аксиальной сегментации нет.':'Совмещение серий не подтверждено. Маска не переносилась.';
      if(im)note+=` ${fmt(im.spacing_mm[0],2)} × ${fmt(im.spacing_mm[1],2)} мм/px.`;
      node.querySelector('.plane-note').textContent=note;
      draw(p);
    }
    renderMetrics();
  }

  async function draw(p){
    const f=frame(p),im=image(p),node=panel(p),canvas=node.querySelector('.image'),wrap=node.querySelector('.canvas-wrap');
    const w=wrap.clientWidth,h=wrap.clientHeight,dpr=Math.min(devicePixelRatio||1,2),token=++drawTokens[p];
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#09121b';ctx.fillRect(0,0,w,h);
    const overview=node.querySelector('.overview');overview.hidden=!state.zoom||!im;
    if(!f||!im){ctx.fillStyle='#b4c9d7';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText('Нет изображений этой плоскости',w/2,h/2);return;}
    let bitmap;try{bitmap=await loadImage(im.png);}catch{ctx.fillStyle='#ffc1a3';ctx.fillText('Изображение не удалось прочитать',16,40);return;}
    if(token!==drawTokens[p])return;
    let sx=0,sy=0,sw=im.width,sh=im.height;
    const entries=planeData(p),representative=entries.frames[entries.selected_index||0];
    const focus=f.candidate_mask_pixels?f.focus_xy:representative?.focus_xy||f.focus_xy;
    if(state.zoom&&focus&&(representative?.candidate_mask_pixels||f.candidate_mask_pixels)){
      const radius=data.synthetic?64:state.radius;
      sw=Math.min(im.width,2*radius);sh=Math.min(im.height,2*radius);
      sx=Math.max(0,Math.min(im.width-sw,focus[0]+.5-sw/2));sy=Math.max(0,Math.min(im.height-sh,focus[1]+.5-sh/2));
    }
    const row=im.spacing_mm[0],col=im.spacing_mm[1],scale=Math.min((w-20)/(sw*col),(h-24)/(sh*row));
    const dx=scale*col,dy=scale*row,dw=sw*dx,dh=sh*dy,ox=(w-dw)/2,oy=(h-dh)/2;
    const project=xy=>[ox+(xy[0]+.5-sx)*dx,oy+(xy[1]+.5-sy)*dy];
    ctx.save();ctx.beginPath();ctx.rect(ox,oy,dw,dh);ctx.clip();ctx.imageSmoothingEnabled=false;
    ctx.filter='brightness(1.25)';ctx.drawImage(bitmap,sx,sy,sw,sh,ox,oy,dw,dh);ctx.filter='none';
    if(state.mask){
      ctx.beginPath();
      for(const loop of f.mask_polygons_xy||[]){
        if(!loop.length)continue;loop.forEach((xy,i)=>{const q=project(xy);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();
      }
      ctx.fillStyle='rgba(100,255,191,.12)';ctx.fill('evenodd');ctx.strokeStyle='rgba(0,0,0,.8)';ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle='#8bffd0';ctx.lineWidth=2.5;ctx.stroke();
    }
    if(state.reference&&f.reference_line_xy){ctx.beginPath();ctx.moveTo(...project(f.reference_line_xy[0]));ctx.lineTo(...project(f.reference_line_xy[1]));ctx.setLineDash([5,4]);ctx.strokeStyle='#edce8d';ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);}
    const labels=[];
    if(state.measures){
      for(const [key,label,color] of [['base','B · основание','#83c6ff'],['displaced','D · наружная часть','#ffb477'],['depth','H · глубина','#fff']]){
        const segment=f.record?.segments_xy?.[key];if(!segment||segment.length!==2)continue;
        const a=project(segment[0]),b=project(segment[1]);
        ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle='#06111b';ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
        const vx=b[0]-a[0],vy=b[1]-a[1],len=Math.hypot(vx,vy)||1,nx=-vy/len*4,ny=vx/len*4;
        ctx.beginPath();for(const q of [a,b]){ctx.moveTo(q[0]-nx,q[1]-ny);ctx.lineTo(q[0]+nx,q[1]+ny);}ctx.stroke();
        labels.push({key,label,color,point:[(a[0]+b[0])/2,(a[1]+b[1])/2]});
      }
    }
    ctx.restore();
    const used=[];ctx.font='13px system-ui';
    for(const item of labels){
      const lw=ctx.measureText(item.label).width+14,lh=25;
      const candidates=item.key==='base'?[[-lw-18,-35],[-lw-18,15],[18,-50]]:item.key==='displaced'?[[20,-2],[20,-35],[-lw-18,30]]:[[14,28],[-lw-8,36],[18,55]];
      let best=null,bestScore=Infinity;
      for(const [offx,offy] of candidates){
        const x=Math.max(7,Math.min(w-lw-7,item.point[0]+offx)),y=Math.max(100,Math.min(h-lh-10,item.point[1]+offy));
        const overlap=used.reduce((s,r)=>s+(Math.max(0,Math.min(x+lw,r.x+r.w)-Math.max(x,r.x))*Math.max(0,Math.min(y+lh,r.y+r.h)-Math.max(y,r.y))),0);
        const score=overlap*10+Math.hypot(x-item.point[0],y-item.point[1]);if(score<bestScore){best={x,y,w:lw,h:lh};bestScore=score;}
      }
      used.push(best);ctx.beginPath();ctx.moveTo(...item.point);ctx.lineTo(best.x+(best.x>item.point[0]?0:best.w),best.y+lh/2);ctx.strokeStyle=item.color;ctx.lineWidth=1;ctx.stroke();
      ctx.fillStyle='rgba(9,20,30,.94)';ctx.fillRect(best.x,best.y,lw,lh);ctx.strokeStyle='#42545e';ctx.strokeRect(best.x+.5,best.y+.5,lw-1,lh-1);ctx.fillStyle=item.color;ctx.textAlign='left';ctx.fillText(item.label,best.x+7,best.y+17);
    }
    const bar=Math.min(10*scale,w/3);ctx.strokeStyle='#a9bdcb';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(16,h-18);ctx.lineTo(16+bar,h-18);ctx.moveTo(16,h-21);ctx.lineTo(16,h-15);ctx.moveTo(16+bar,h-21);ctx.lineTo(16+bar,h-15);ctx.stroke();ctx.fillStyle='#a9bdcb';ctx.font='11px system-ui';ctx.fillText(fmt(bar/scale,0)+' мм',16,h-28);
    overview.width=156;overview.height=184;const c=overview.getContext('2d');c.setTransform(2,0,0,2,0,0);c.fillStyle='#03090e';c.fillRect(0,0,78,92);
    const small=Math.min(78/(im.width*col),92/(im.height*row)),ix=(78-im.width*col*small)/2,iy=(92-im.height*row*small)/2;
    c.drawImage(bitmap,ix,iy,im.width*col*small,im.height*row*small);c.strokeStyle='#8bffd0';c.lineWidth=1;c.strokeRect(ix+sx*col*small,iy+sy*row*small,sw*col*small,sh*row*small);
  }

  function renderMetrics(){
    const f=frame(state.plane),r=f?.record||{},projected=!data.synthetic&&state.plane==='axial';
    const b=r.candidate_base_width_mm,d=r.candidate_displaced_width_mm,h=r.candidate_depth_mm,ratio=r.candidate_ratio,u=r.uncertainty_mm;
    text('#metric-b',fmt(b));text('#metric-d',fmt(d));text('#metric-h',fmt(h));text('#metric-r',fmt(ratio));
    text('#metric-source',`${state.plane==='sagittal'?'Сагиттальная':'Аксиальная'} · срез ${(state.indices[state.plane]||0)+1}${data.synthetic?'':' · предварительно'}`);
    text('#ratio-formula',typeof b==='number'&&b>0&&typeof d==='number'?`D/B = ${fmt(d)} / ${fmt(b)} = ${fmt(ratio)}`:'D/B не определено: нет пригодного основания.');
    $$('.plane-tabs button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.metricPlane===state.plane)));
    let conclusion='Недостаточно данных';
    if(projected)conclusion='Только перенос маски';
    else if(r.geometry_measurable&&r.geometric_candidate==='extrusion')conclusion='Похоже на экструзию';
    else if(r.geometry_measurable&&r.geometric_candidate==='protrusion')conclusion='Похоже на протрузию';
    text('#frame-decision',conclusion);
    if([b,d,u].every(x=>typeof x==='number'&&Number.isFinite(x))){
      const delta=Math.abs(d-b),threshold=2*u;
      text('#difference',`${d>=b?'D − B':'B − D'} = ${fmt(delta,3)} мм ${delta>threshold?'>':'≤'} 2u = ${fmt(threshold,3)} мм`);
      text('#uncertainty',`u = ${fmt(u,3)} мм — нижняя оценка по разрешению. Порог 2u относится к разнице ширин, а не к глубине выпячивания.`);
    }else{text('#difference','Пригодного сравнения B и D нет.');text('#uncertainty','Отсутствие измерения не означает отсутствие грыжи.');}
    let explanation=data.synthetic?'Известная учебная маска; решение рассчитано тем же модулем геометрии.':projected?'Проекция не даёт второй независимой оценки формы. Тип по этому аксиальному контуру не подтверждается.':r.geometry_measurable?'Это геометрия кандидатной маски относительно приближённой границы. Анатомия и контур требуют проверки.':r.quality_flags?.includes('empty_mask')?'На этом срезе нет кандидатной маски. Листайте соседние срезы; отсутствие маски не исключает грыжу.':'Числа описывают маску, но данных для определения её формы недостаточно.';
    const reasons={disconnected_mask:'Маска состоит из несвязных частей.',multiple_base_intervals:'Пересечение опорной линии неоднозначно.',base_not_intersecting_reference:'Основание не пересекает опорную линию.',missing_base:'Пригодное основание не найдено.',no_resolved_displaced_tissue:'Наружная часть не разрешается относительно этой границы.',touches_frame_edge:'Маска касается края кадра.'};
    if(!data.synthetic&&!r.geometry_measurable)for(const flag of r.quality_flags||[])if(reasons[flag])explanation+=' '+reasons[flag];
    text('#frame-explanation',explanation);
    text('#clinical-result',data.synthetic?'Не применяется — учебная схема':'Не определён');
    text('#overall-hypothesis','Гипотеза по выбранным представительным срезам: '+(names[current().decision?.geometric_candidate]||'Не определено'));
    text('#record',JSON.stringify({image_id:f?.image_id,mask_source:f?.mask_source,...r},null,2));
  }

  if(!data.cases?.length){text('h1','В отчёте нет результатов');return;}
  text('#study',data.study_id);text('#source',data.synthetic?'Учебная геометрия без нейросети':data.model?.inference_performed?'SpineSegPlus · автоматический запуск':'Внешняя маска · без нового запуска нейросети');
  text('#version',data.tool_version);$('#demo-banner').hidden=!data.synthetic;
  data.cases.forEach((c,index)=>{const button=document.createElement('button');button.type='button';button.setAttribute('aria-pressed',String(index===state.caseIndex));const name=document.createElement('span');name.textContent=c.level;const sub=document.createElement('small');sub.textContent=data.synthetic?'DEMO':c.candidate_voxels?'есть контур':'нет маски';button.append(name,sub);button.addEventListener('click',()=>selectCase(index));$('#levels').append(button);});
  for(const p of ['sagittal','axial']){
    const node=panel(p);
    node.querySelector('.slice').addEventListener('input',event=>setFrame(p,Number(event.target.value)));
    node.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>setFrame(p,(state.indices[p]||0)+Number(button.dataset.step))));
    node.querySelector('.image').addEventListener('click',()=>{state.plane=p;render();});
    node.querySelector('.image').addEventListener('keydown',event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();setFrame(p,(state.indices[p]||0)+(event.key==='ArrowLeft'?-1:1));}});
  }
  $$('.plane-tabs button').forEach(button=>button.addEventListener('click',()=>{state.plane=button.dataset.metricPlane;render();}));
  for(const [selector,key] of [['#layer-mask','mask'],['#layer-reference','reference'],['#layer-measures','measures']])$(selector).addEventListener('change',event=>{state[key]=event.target.checked;render();});
  $('#zoom').addEventListener('click',()=>{state.zoom=!state.zoom;$('#zoom').textContent=state.zoom?'Весь кадр':'Крупно';$('#zoom').setAttribute('aria-pressed',String(state.zoom));$('#zoom-range').disabled=!state.zoom;render();});
  $('#zoom-range').addEventListener('input',event=>{state.radius=Number(event.target.value);render();});
  $('#print').addEventListener('click',()=>window.print());
  $('#export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='disc-morphology-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  new ResizeObserver(()=>{draw('sagittal');draw('axial');}).observe($('.viewer'));
  selectCase(Math.min(state.caseIndex,data.cases.length-1));
})();
