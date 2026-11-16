// Read-only policy diagnostics; always use the shop BEFORE the recorded action.
(() => {
  const section = document.createElement('section');
  section.id = 'buyPetPreferences';
  section.className = 'buy-pet-preferences hidden';
  section.setAttribute('aria-label', 'Which pet the policy would buy');
  $('diagnostics').append(section);
  const style = document.createElement('style');
  style.textContent = `
    .buy-pet-preferences{grid-column:1/-1;border-top:1px solid var(--line);padding-top:14px;margin-top:4px}
    .buy-pet-preferences h3{font-size:17px;margin:0 0 4px}
    .buy-pet-caption,.buy-pet-note{font-size:12px;color:var(--muted);margin:4px 0 10px}
    .buy-pet-option,.buy-pet-columns{display:grid;grid-template-columns:42px minmax(140px,1fr) minmax(100px,2fr) 90px 90px;gap:12px;align-items:center;padding:7px 9px}
    .buy-pet-columns{font-size:11px;color:var(--muted);padding-top:0;padding-bottom:3px}
    .buy-pet-option{border-radius:8px;margin:3px 0;background:var(--panel2)}
    .buy-pet-option.is-chosen{outline:1px solid var(--blue)}
    .buy-pet-option img{width:40px;height:40px;object-fit:contain;transform:scaleX(-1)}
    .buy-pet-option .pet-meta{font-size:11px;color:var(--muted);display:block}
    .buy-pet-option .chosen-label{font-size:10px;color:var(--blue);margin-left:7px}
    .buy-pet-option .bar{height:9px}
    .buy-pet-option .bar i{display:block;height:100%;background:var(--blue)}
    .buy-pet-option .percent,.buy-pet-columns .percent{text-align:right;font-variant-numeric:tabular-nums}
    .buy-pet-option .overall{font-size:12px;color:var(--muted)}
    @media(max-width:700px){.buy-pet-option,.buy-pet-columns{grid-template-columns:36px minmax(100px,1fr) 76px 76px;gap:7px}.buy-pet-option .bar,.buy-pet-columns .bar-heading{display:none}}
  `;
  document.head.append(style);
  function percentage(p) {
    if (p > 0 && p < .001) return '<0.1%';
    if (p < 1 && p > .9995) return '>99.9%';
    return (100*p).toFixed(1)+'%';
  }
  function drawPetPreferences() {
    const frame=frames[index],data=frame.policy?.buyPetSources;
    section.classList.toggle('hidden',frame.phase==='battle'||!data);
    if (frame.phase==='battle'||!data) return;
    const before=frame.before,shop=before?.player?.shop;
    if (data.schema!=='buy-pet-sources-v1'||!Array.isArray(shop)) {
      section.textContent='Pet preferences unavailable for this recorded state.';return;
    }
    if (!data.available) {
      section.innerHTML='<h3>Which pet?</h3><p class="buy-pet-note">Buying a pet was not legal before this action.</p>';return;
    }
    const rows=data.sources.map(r=>({...r,slot:shop.find(s=>s.index===r.source)}));
    if (rows.some(r=>r.slot?.item?.kind!=='pet'||!Number.isFinite(r.conditionalProbability)||!Number.isFinite(r.overallProbability))) {
      section.textContent='Pet preferences do not match the recorded before-action shop.';return;
    }
    rows.sort((a,b)=>b.conditionalProbability-a.conditionalProbability||a.source-b.source);
    section.innerHTML=`<h3>Which pet would it buy?</h3>
      <p class="buy-pet-caption">Before-action shop · ${esc(before.player.gold)} gold · Buy Pet overall: ${esc(percentage(data.typeProbability))}.
      “If buying” compares pets after choosing Buy Pet, across all legal destinations.</p>
      <div class="buy-pet-columns"><span></span><span>Pet / original shop slot</span><span class="bar-heading"></span><span class="percent">If buying</span><span class="percent">Overall</span></div>
      ${rows.map(r=>{const pet=r.slot.item,chosen=frame.action?.type==='BUY_PET'&&frame.action.source===r.source;return `
        <div class="buy-pet-option${chosen?' is-chosen':''}" data-source="${r.source}">
          <img src="${spritePath(pet)}" alt="" aria-hidden="true">
          <span><strong>${esc(pet.label)}</strong>${chosen?'<span class="chosen-label">Chosen</span>':''}
            <span class="pet-meta">Shop ${r.source} · ${pet.attack}/${pet.health} · L${pet.level}+${pet.experience} · ${r.slot.cost}g</span></span>
          <span class="bar"><i style="width:${Math.max(0,Math.min(100,100*r.conditionalProbability))}%"></i></span>
          <strong class="percent conditional">${esc(percentage(r.conditionalProbability))}</strong>
          <span class="percent overall">${esc(percentage(r.overallProbability))}</span>
        </div>`}).join('')}
      <p class="buy-pet-note">Overall = Buy Pet probability × pet probability. These are policy preferences, not win chances; the replay chooses deterministically.</p>`;
  }
  const previousRender=render;
  render=function(){previousRender();drawPetPreferences();};
  render();
})();
