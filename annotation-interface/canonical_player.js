/* One annotation clock, independent of each MP4's frame-rounded duration.
 * All existing annotation controls use this adapter; the DOM element decodes only.
 * A chunk's measured media duration maps linearly onto its canonical interval.
 */
function createAnnotationPlayer(media) {
  let timeline=null,ranges=[],active=null,target=0,pending=false,wantsPlay=false,waiting=false;
  function seek(t) {
    target=Math.max(0,Math.min(timeline.play_duration_seconds,Number(t)||0));
    const r=ranges.find(r=>target>=r.start&&target<r.end)||
      (target===timeline.play_duration_seconds?ranges.find(r=>r.end===target):null);
    if(!r){media.pause();active=null;pending=true;waiting=true;return;}
    waiting=false;
    if(!active||active.url!==r.url){active=r;pending=true;media.pause();media.src=r.url;media.load();}
    else {active=r;if(media.readyState>=1)applySeek();}
  }
  function applySeek(){
    if(!active||!Number.isFinite(media.duration)||media.duration<=0)return;
    media.currentTime=(target-active.start)/(active.end-active.start)*media.duration;
    pending=false;
    if(wantsPlay)media.play().catch(()=>{wantsPlay=false;});
  }
  const methods={
    setTimeline(meta){methods.resetTimeline();timeline={...meta,play_duration_seconds:meta.canonical_duration_seconds??meta.play_duration_seconds};},
    resetTimeline(){media.pause();timeline=null;ranges=[];active=null;target=0;pending=false;waiting=false;wantsPlay=false;},
    setRanges(next){ranges=next;if(timeline&&(waiting||!active))seek(target);},
    play(){if(!timeline)return media.play();wantsPlay=true;if(target>=timeline.play_duration_seconds)seek(0);if(waiting||pending)return Promise.resolve();return media.play();},
    pause(){wantsPlay=false;media.pause();}
  };
  media.addEventListener('loadedmetadata',()=>{if(timeline&&pending)applySeek();});
  media.addEventListener('ended',()=>{
    if(!timeline||!active)return;
    target=active.end;
    if(target<timeline.play_duration_seconds){seek(target);}
    else {wantsPlay=false;}
  });
  return new Proxy(media,{
    get(obj,key){
      if(key in methods)return methods[key];
      if(key==='waitingForCache')return waiting;
      if(timeline){
        if(key==='duration')return timeline.play_duration_seconds;
        if(key==='currentTime')return pending||!active?target:active.start+Math.min(1,media.currentTime/media.duration)*(active.end-active.start);
        if(key==='paused')return !wantsPlay;
        if(key==='ended')return target>=timeline.play_duration_seconds&&!wantsPlay;
      }
      const value=Reflect.get(obj,key,obj);return typeof value==='function'?value.bind(obj):value;
    },
    set(obj,key,value){if(timeline&&key==='currentTime'){seek(value);return true;}return Reflect.set(obj,key,value,obj);}
  });
}
