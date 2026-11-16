// Recorded simulator snapshots only: no combat rules or random draws in JS.
(() => {
  let eventIndex = 0, frameIndex = -1, timer = null;
  const panel = document.createElement('section');
  panel.className = 'controls hidden';
  panel.style.cssText = 'padding:12px;background:var(--panel2);border-radius:10px;margin:12px 0';
  panel.innerHTML = `<button id="battleBack" aria-label="Previous battle event">◀ Event</button>
    <button id="battlePlay">▶ Play battle</button><button id="battleNext">Event ▶</button>
    <label>Speed <select id="battleSpeed"><option value="1600">0.5×</option><option value="800" selected>1×</option><option value="400">2×</option><option value="200">4×</option></select></label>
    <input id="battleSeek" aria-label="Battle event" type="range" min="0" value="0" style="flex:1;min-width:140px">
    <span id="battleCounter"></span><button id="battleFlag">Save this moment</button>
    <div id="battleEvent" style="width:100%;font-size:18px;font-weight:700"></div>
    <small>Recorded trigger-log / phase boundaries. A phase can contain multiple effects. Damage and perk changes below are observed state differences.</small>`;
  $('battleResult').after(panel);
  const style = document.createElement('style');
  style.textContent = '@keyframes battlePulse{from{filter:brightness(1.6);transform:translateY(-5px)}to{filter:none;transform:none}}@keyframes clashRight{50%{transform:translateX(18px)}}@keyframes clashLeft{50%{transform:translateX(-18px)}}.battle-pulse{animation:battlePulse .4s ease-out}.clash-right{animation:clashRight .35s ease-in-out}.clash-left{animation:clashLeft .35s ease-in-out}@media(prefers-reduced-motion:reduce){.battle-pulse,.clash-right,.clash-left{animation:none}}';
  document.head.append(style);
  const events = () => frames[index]?.battle?.playback?.events || [];
  function stop() { clearInterval(timer); timer=null; $('battlePlay').textContent='▶ Play battle'; }
  function draw() {
    const list=events(); if(!list.length)return;
    eventIndex=Math.max(0,Math.min(eventIndex,list.length-1));
    const current=list[eventIndex],previous=list[Math.max(0,eventIndex-1)];
    const side=trace.metadata.perspective==='p1'?1:0;
    $('team').innerHTML=teamCards(current.teams[side],previous.teams[side],null,true);
    $('opponent').innerHTML=teamCards(current.teams[1-side],previous.teams[1-side],null,false);
    panel.parentElement.querySelectorAll('.card.changed').forEach(card=>card.classList.add('battle-pulse'));
    if(current.kind==='phase'&&current.label==='Attack'){
      $('team').lastElementChild?.classList.add('clash-right');
      $('opponent').firstElementChild?.classList.add('clash-left');
    }
    // Legacy battle frames stored the next shop's score. Do not show next
    // turn's gold/hearts as if they were the battle's starting scoreboard.
    const f=frames[index];
    const prior=frames.slice(0,index).reverse().find(x=>x.turn===f.turn&&x.phase==='shop');
    const pre=prior?.after?.player?.turn===f.turn?prior.after:prior?.before;
    const hud=eventIndex===list.length-1?f.after:(pre||f.before);
    if(hud){$('score').innerHTML=score(hud);$('opponentScore').innerHTML=opponentScore(hud);}
    $('actionMeta').textContent=`Turn ${f.turn} · ${current.kind==='trigger'?'trigger log':'phase snapshot'}`;
    $('battleCounter').textContent=`${eventIndex+1} / ${list.length}`;
    $('battleEvent').textContent=current.label;
    $('battleSeek').max=list.length-1; $('battleSeek').value=eventIndex;
    $('battleBack').disabled=eventIndex===0; $('battleNext').disabled=eventIndex===list.length-1;
    $('battleResult').textContent=eventIndex===list.length-1?frames[index].battle.resultLabel:'Battle playback';
    const changes=[];
    for(let s=0;s<2;s++)for(let i=0;i<5;i++){
      const a=previous.teams[s][i],b=current.teams[s][i],who=s===side?'You':'Opponent';
      if(itemSig(a)===itemSig(b))continue;
      const describe=p=>p?`${p.label} ${p.attack}/${p.health}${p.statusLabel?' · '+p.statusLabel:''}`:'empty';
      changes.push(`${who}, slot ${i}: ${describe(a)} → ${describe(b)}`);
    }
    $('changes').innerHTML=(changes.length?changes:['No visible board change at this boundary']).map(s=>`<span class="change">${esc(s)}</span>`).join('');
  }
  const oldRender=render;
  render=function(){
    if(frameIndex!==index){stop();eventIndex=0;frameIndex=index;}
    oldRender();panel.classList.toggle('hidden',!events().length);if(events().length)draw();
  };
  $('battleBack').onclick=()=>{stop();eventIndex--;draw();};
  $('battleNext').onclick=()=>{stop();eventIndex++;draw();};
  $('battleSeek').oninput=e=>{stop();eventIndex=Number(e.target.value);draw();};
  $('battlePlay').onclick=()=>{
    if(timer){stop();return;}
    if(eventIndex===events().length-1)eventIndex=0;
    draw();$('battlePlay').textContent='❚❚ Pause';
    timer=setInterval(()=>{eventIndex++;draw();if(eventIndex>=events().length-1)stop();},Number($('battleSpeed').value));
  };
  $('battleSpeed').onchange=()=>{if(timer){stop();$('battlePlay').click();}};
  $('battleFlag').onclick=()=>{
    const data={schema:'flagged-battle-moment-v1',metadata:trace.metadata,frame:index,event:eventIndex,
      turn:frames[index].turn,playback:frames[index].battle.playback};
    const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    link.href=url;link.download=`sap-seed-${trace.metadata.seed}-turn-${frames[index].turn}-event-${eventIndex}.json`;
    link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  render();
})();
