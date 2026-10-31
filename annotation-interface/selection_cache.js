/* Selection owns the single annotation player. Polls only observe/renew leases. */
(()=>{
  const fields=['trial','date','camera'].map(id=>document.getElementById(id));
  const client=Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join(''), panel=document.createElement('section');
  panel.id='selectionCache';panel.style.cssText='margin:12px 16px;padding:12px;border:1px solid #475569;border-radius:10px;background:#151e2c';
  const status=document.createElement('div'),ranges=document.createElement('div'),note=document.createElement('div');
  status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  ranges.style.cssText='display:flex;gap:2px;min-height:24px;margin:8px 0';
  ranges.setAttribute('aria-label','Playable cached annotation ranges');
  note.textContent='Green = playable; gray = waiting for cache. Click a green range or seek using the annotation timeline. Uncached seeks wait without skipping time. Annotation times use the canonical wall-clock timeline, not rounded chunk durations.';
  panel.append(status,ranges,note);document.querySelector('header').after(panel);
  let changeAttempt=0,key=null,revision=0,debounce=null,poll=null,ready=[],installed=null,cacheState='idle';
  async function request(url,options){const response=await fetch(url,options);if(!response.ok)throw new Error('Cache request failed: '+response.status);return response.json()}
  function release(rev){if(rev>0)navigator.sendBeacon('/api/selection-release',JSON.stringify({client,revision:rev}));}
  async function install(meta,mine){
    if(installed===meta.video_name)return;
    installed=meta.video_name;
    currentMeta=meta;currentVideoName=meta.video_name;
    video.resetTimeline();
    if(meta.playback_mode==='chunks-v1')video.setTimeline(meta);
    else {video.src=meta.video_url;video.load();}
    // Do not enable editing until the pinned annotation snapshot has arrived.
    annotationLocked=true;
    await loadAnnotations();
    if(mine!==revision)return;
    video.style.display='block';videoControls.style.display='flex';ph.style.display='none';ethowrap.style.display='block';
    drawEthogram();
  }
  async function render(data,mine){
    cacheState=data.state;
    if(data.meta)await install(data.meta,mine);
    if(mine!==revision)return;
    ready=data.ready||[];
    if(data.meta?.playback_mode==='chunks-v1')video.setRanges(ready);
    const estimate=data.state==='queued'?'Wait estimate unknown (queued)':data.eta_seconds==null?'Wait estimate unknown (measuring completed chunks)':data.state==='done'?'Ready':`Estimated remaining ~${Math.ceil(data.eta_seconds)} s (observed throughput; may vary)`;
    status.textContent=data.meta?.playback_mode==='file'?'Using existing full annotation video; no rendering queued.':`Selected camera: ${data.state} · ${ready.length}/${data.total||'?'} chunks playable · ${estimate}${data.error?' · '+data.error:''}`;
    ranges.replaceChildren();let cursor=0,total=data.duration||1;
    function band(start,end,range){if(end<=start)return;const el=document.createElement(range?'button':'span');el.style.cssText=`flex:0 0 ${(end-start)/total*100}%;min-width:0;padding:0;border-radius:2px;background:${range?'#15803d':'#374151'};height:24px`;el.title=`${start.toFixed(3)}–${end.toFixed(3)} canonical s · ${range?'cached/playable '+range.kind:'not cached'}`;el.setAttribute('aria-label',el.title);if(range)el.addEventListener('click',()=>{video.currentTime=start;video.play().catch(()=>{});drawEthogram()});ranges.append(el)}
    if(data.meta?.playback_mode!=='file'){for(const r of ready){band(cursor,r.start,null);band(r.start,r.end,r);cursor=r.end}band(cursor,total,null);}
  }
  async function check(mine){if(mine!==revision)return;try{const data=await request('/api/selection-cache?'+new URLSearchParams({client,revision:mine}));if(mine!==revision||data.revision!==mine)return;await render(data,mine);if(['queued','running'].includes(data.state))poll=setTimeout(()=>check(mine),1000)}catch(e){if(mine===revision){status.textContent=e.message+'; retrying status';poll=setTimeout(()=>check(mine),3000)}}}
  async function changed(force=false){
    const attempt=++changeAttempt;
    const values=fields.map(el=>el.value),next=values.join('|');if(next===key&&!force)return;const same=next===key;
    if(saveTimer||saveInFlight||saveDirty){
      status.textContent=`Saving loaded camera ${currentMeta?.camera||''} before switching to ${values[2]}…`;
      const saved=await saveAnnotations();
      if(attempt!==changeAttempt||next!==fields.map(el=>el.value).join('|'))return;
      if(!saved){status.textContent=annstatus.textContent;return;}
    }
    if(!same&&currentMeta&&!clearLoadedVideo())return;
    release(revision);key=next;const mine=++revision;
    clearTimeout(debounce);clearTimeout(poll);ready=[];if(!same)installed=null;ranges.replaceChildren();
    if(!values.every(Boolean)){status.textContent='Select a trial, day and camera.';return;}
    status.textContent='Waiting for selected camera cache…';
    debounce=setTimeout(async()=>{if(mine!==revision)return;try{const data=await request('/api/selection-cache',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client,revision:mine,trial:values[0],date:values[1],camera:values[2]})});if(mine!==revision||data.revision!==mine)return;await render(data,mine);if(['queued','running'].includes(data.state))poll=setTimeout(()=>check(mine),1000)}catch(e){if(mine===revision)status.textContent=e.message+'; use Load / retry'}},350);
  }
  window.requestSelectedPlayback=()=>{if(fields.map(el=>el.value).join('|')===key&&currentMeta&&installed&&!saveDirty&&!saveInFlight&&!['error','cancelled'].includes(cacheState))return;return changed(true);};
  window.addEventListener('pagehide',()=>release(revision));
  window.addEventListener('pageshow',e=>{if(e.persisted){key=null;changed();}});
  for(const field of fields){field.addEventListener('change',()=>queueMicrotask(changed));new MutationObserver(()=>queueMicrotask(changed)).observe(field,{childList:true})}
  changed();
})();
