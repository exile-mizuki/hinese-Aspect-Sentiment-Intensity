'use strict';
/* Guangan V4.2 — calm Fernway-style UI; the frozen /api/predict endpoint remains the single source of live results. */
const $ = id => document.getElementById(id);
const svgns = 'http://www.w3.org/2000/svg';
const palette = ['#7565ed','#35b9ae','#bd8bca','#4b9ab8','#e1a58a','#6b9aa3','#ae9acb','#d4a87f'];
const labels = {overview:'总览',predict:'情绪预测',data:'数据洞察',experiments:'模型实验',story:'研究展示',about:'项目与复现'};
const examples = [
  {text:'汤非常难喝，但是服务很好。',aspects:['汤','服务']},
  {text:'这次的牛肉面非常好吃，汤也很鲜美。',aspects:['牛肉面','汤']},
  {text:'等了很久才上菜，服务让我有点失望。',aspects:['服务']}
];
const ambiguity = [
  {label:'单观点',n:6562,value:.304641},
  {label:'多观点',n:914,value:.314595},
  {label:'正负混合',n:87,value:.464670,strong:true},
  {label:'观点差 ≥ 1',n:218,value:.381890},
  {label:'观点差 ≥ 2',n:52,value:.519920,highlight:true}
];
const expGroups = {
  seeds:{note:'训练集 Group CV · 3 种子平均 · 不是独立测试',title:'三随机种子：综合 MAE',items:[
    {name:'Opinion-Aware v2',value:.304790},
    {name:'Arousal MSE',value:.304849},
    {name:'Aux 0.05',value:.305099},
    {name:'Arousal Huber',value:.305430},
    {name:'原始 BERT L1',value:.305531,baseline:true},
    {name:'EmoBank E2',value:.305736}
  ],insight:'三种子平均的改善幅度不足 0.001，属于微弱收益；不能仅据此认定新方法有更好的泛化。'},
  dev:{note:'已有官方 Validation · 115 方面 · 多次查看过，非全新盲测',title:'官方 Dev：已有验收结果',items:[
    {name:'原始 BERT L1',value:.306696,baseline:true},
    {name:'Opinion-Aware v2',value:.311478},
    {name:'Arousal MSE',value:.316261},
    {name:'Aux 0.05',value:.317043}
  ],insight:'在已有官方 Dev 验收中，原始 BERT L1 的综合 MAE 最低。Dev 经多次查看，不应当作未使用过的独立测试。'},
  single:{note:'训练集 Group CV · Seed 42 · 探索性模型比较',title:'单随机种子：探索性结果',items:[
    {name:'Arousal MSE',value:.304255},
    {name:'Opinion-Aware v2',value:.304813},
    {name:'Aspect Pooling',value:.304937},
    {name:'Aspect Local',value:.305230},
    {name:'原始 BERT L1',value:.305858,baseline:true}
  ],insight:'Aspect Pooling 与 Aspect Local 目前缺少三种子或独立验收证据，不能因单种子改善就替换正式模型。'}
};
const storySlides = [
  {chapter:'01 / RESEARCH QUESTION',title:'同一句评论，<em>不止一个判断。</em>',copy:'整句情绪分析难以区分“服务很好”和“价格偏贵”。本项目预测指定方面的 V（效价）与 A（唤醒度）。',stats:[['02','独立评分维度'],['01–09','连续取值范围'],['ASPECT','任务颗粒度']]},
  {chapter:'02 / DATA & LABELS',title:'8,523 条观点，<em>7,476 个方面。</em>',copy:'同一方面可能包含多个原始观点。按照任务定义聚合均值，同时审计意见冲突、标注差异与标签稀疏性。',stats:[['6,050','评论句子'],['914','多观点方面'],['87','正负观点混合']]},
  {chapter:'03 / EXPERIMENT',title:'性能提升，<em>必须接受验证。</em>',copy:'冻结按评论分组的三折划分，在同条件下比较不同模型，并用已有 Validation 核对最终模型的泛化表现。',stats:[['40','完整实验组'],['03','历史重复预测副本'],['0.3067','正式模型 Dev MAE']]},
  {chapter:'04 / CONCLUSION',title:'保持克制，<em>让证据说话。</em>',copy:'Opinion-Aware 在部分混合情绪样本上改善误差，但 Dev 未超过基线。正式交付继续采用 BERT 方面标记 L1 模型。',stats:[['0.4647','混合样本基线 MAE'],['0.4432','Opinion-Aware MAE'],['EPOCH 3','交付模型版本']]}
];
let meta=null,ready=false,busy=false,current=null,history=[],activePoint=0,storyIndex=0,activePage='overview',toastTimer=null;
const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt=n=>Number(n).toFixed(2);
const cleanAspects=()=>[...new Set($('aspects').value.split(/[,，;；\n]+/).map(a=>a.trim()).filter(Boolean))];
function create(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=String(text);return node}
function svgEl(tag,attrs={},text){const n=document.createElementNS(svgns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!==undefined)n.textContent=String(text);return n}
function notify(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,2600)}
function showError(message){$('error').textContent=message||'';$('error').hidden=!message}
function navigate(page){window.location.hash='#/'+page}
function route(){let requested=decodeURIComponent(location.hash.slice(2).split(/[/?]/)[0]||'overview');if(!labels[requested])requested='overview';activePage=requested;
  document.querySelectorAll('[data-page]').forEach(p=>p.hidden=p.dataset.page!==requested);
  document.querySelectorAll('[data-nav]').forEach(a=>{const on=a.dataset.nav===requested;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
  $('page-breadcrumb').textContent=labels[requested];document.title=labels[requested]+' · 观感';closeMenu();
  const view=document.querySelector('[data-page=\"'+requested+'\"]');
  if(view&&!reduced()&&view.animate)view.animate([{opacity:.36,transform:'translateY(9px)'},{opacity:1,transform:'translateY(0px)'}],{duration:340,easing:'cubic-bezier(.19,.87,.32,1)'});
  if(requested==='story')renderStory({reveal:true});if(requested==='experiments')renderExperiments();if(requested==='overview')playOverviewHeroIntro();
  scrollTo({top:0,behavior:'instant'});
}
function closeMenu(){$('sidebar').classList.remove('open');$('sidebar-backdrop').hidden=true;$('menu-toggle').setAttribute('aria-expanded','false')}
function toggleMenu(){let on=!$('sidebar').classList.contains('open');$('sidebar').classList.toggle('open',on);$('sidebar-backdrop').hidden=!on;$('menu-toggle').setAttribute('aria-expanded',String(on))}
let staticPreviewData=null;
async function getJSON(path,options){
 if(!staticPreviewData){const response=await fetch('./preview-data.json');if(!response.ok)throw Error('静态示例文件缺失');staticPreviewData=await response.json()}
 if(path==='/api/status')return {state:'ready',message:'静态预览 · 三条真实保存示例'};
 if(path==='/api/meta')return staticPreviewData.metadata;
 if(path==='/api/predict'){
  const input=JSON.parse(options.body);
  const saved=staticPreviewData.samples.find(x=>x.text===input.text&&JSON.stringify(x.aspects)===JSON.stringify(input.aspects));
  if(!saved)throw Error('静态预览仅支持页面提供的三条原样示例；任意评论请启动本地模型。');
  const result=JSON.parse(JSON.stringify(saved));
  result.compare=input.compare;
  if(!input.compare)for(const p of result.predictions){delete p.candidate_V;delete p.candidate_A;delete p.delta_V;delete p.delta_A}
  return result;
 }
 throw Error('静态预览不支持此请求');
}
function updateStatus(state,message){ready=state==='ready';for(const id of ['side-dot','top-dot']){$(id).className='live-dot '+(ready?'ready':state==='error'?'error-state':'')};
 $('side-state-title').textContent=ready?'模型已就绪':state==='error'?'连接异常':'加载模型中';$('side-state-desc').textContent=ready?'SAVED EXAMPLES · PREVIEW':'STATIC PREVIEW';$('top-status').textContent=message||'正在连接';updateControls()}
function updateControls(){$('analyze').disabled=!ready||busy||!meta;$('analyze-label').textContent=busy?'模型正在推理…':ready?'运行情绪分析':'等待模型加载';$('submit-spinner').hidden=!busy}
async function poll(){try{let s=await getJSON('/api/status');updateStatus(s.state,s.message)}catch{updateStatus('error','服务断开，请启动本地后端')}setTimeout(poll,ready?10000:2000)}
async function loadMeta(){try{meta=await getJSON('/api/meta');const info=meta.current_model||{};if(info.checkpoint_sha256)$('model-hash').textContent=String(info.checkpoint_sha256).slice(0,20)+'…';else $('model-hash').textContent='见 release_v1/baseline_manifest.json';if(info.id&& !String(info.id).includes('epoch3'))notify('检测到旧版模型标识，请确认后端版本');updateControls()}catch(e){showError('模型元数据尚不可用：'+e.message)}}
function syncInput(){$('count').textContent=Array.from($('comment').value).length+' / 500';const c=$('aspect-chips');c.replaceChildren(...cleanAspects().map(a=>create('span','tag',a)));
 if(current){const changed=current.text!==$('comment').value.trim()||JSON.stringify(current.aspects)!==JSON.stringify(cleanAspects())||current.compare!==$('compare').checked;$('stale-notice').hidden=!changed}}
function preset(i){const p=examples[i];$('comment').value=p.text;$('aspects').value=p.aspects.join('，');showError('');syncInput()}
function clip(v){return Math.max(1,Math.min(9,Number(v)))}
function renderTextHighlights(text,aspects){const out=$('submitted-text');out.replaceChildren();const terms=aspects.filter(t=>t!=='NULL').sort((a,b)=>b.length-a.length);let pos=0,plain='';
 while(pos<text.length){const hit=terms.find(t=>text.startsWith(t,pos));if(hit){if(plain){out.append(document.createTextNode(plain));plain=''}out.append(create('mark','',hit));pos+=hit.length}else{const c=String.fromCodePoint(text.codePointAt(pos));plain+=c;pos+=c.length}}if(plain)out.append(document.createTextNode(plain))}
function selectPoint(index){activePoint=index;document.querySelectorAll('[data-score-index]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.scoreIndex)===index));document.querySelectorAll('[data-plot-index]').forEach(n=>n.classList.toggle('selected',Number(n.dataset.plotIndex)===index));document.querySelectorAll('[data-legend-index]').forEach(n=>n.classList.toggle('selected',Number(n.dataset.legendIndex)===index))}
function renderScatter(result){const graph=$('result-chart');graph.replaceChildren();const svg=svgEl('svg',{viewBox:'0 0 700 430',role:'img','aria-label':'目标方面的 V/A 情绪坐标图。准确数字见下方评分卡片'});const plot={x:60,y:38,w:595,h:340};const x=v=>plot.x+(clip(v)-1)/8*plot.w,y=a=>plot.y+plot.h-(clip(a)-1)/8*plot.h;
 svg.append(svgEl('rect',{x:plot.x,y:plot.y,width:plot.w,height:plot.h,rx:5,fill:'#fbfaff'}));
 for(const tick of [1,3,5,7,9]){const dashed=tick===5;svg.append(svgEl('path',{d:`M${x(tick)} ${plot.y}V${plot.y+plot.h} M${plot.x} ${y(tick)}H${plot.x+plot.w}`,stroke:dashed?'#b7b3d5':'#e7e5ee','stroke-dasharray':dashed?'4 5':'0',fill:'none'}));svg.append(svgEl('text',{x:x(tick),y:plot.y+plot.h+22,fill:'#8585a2','font-size':12,'text-anchor':'middle'},tick));svg.append(svgEl('text',{x:plot.x-17,y:y(tick)+4,fill:'#8585a2','font-size':12,'text-anchor':'end'},tick))}
 svg.append(svgEl('text',{x:plot.x,y:22,fill:'#8188a0','font-size':12},'A · 情绪强烈程度'));svg.append(svgEl('text',{x:plot.x+plot.w,y:421,fill:'#8188a0','font-size':12,'text-anchor':'end'},'V · 情感正负方向'));
 result.predictions.forEach((p,i)=>{const color=palette[i%palette.length],group=svgEl('g',{'data-plot-index':i});if(result.compare&&Number.isFinite(Number(p.candidate_V))&&Number.isFinite(Number(p.candidate_A))){group.append(svgEl('line',{x1:x(p.V),y1:y(p.A),x2:x(p.candidate_V),y2:y(p.candidate_A),stroke:color,'stroke-width':2,'stroke-dasharray':'5 4',opacity:.45}));group.append(svgEl('rect',{x:x(p.candidate_V)-5,y:y(p.candidate_A)-5,width:10,height:10,transform:`rotate(45 ${x(p.candidate_V)} ${y(p.candidate_A)})`,fill:'white',stroke:color,'stroke-width':2}))}
 group.append(svgEl('circle',{cx:x(p.V),cy:y(p.A),r:19,fill:color,opacity:.12}));const point=svgEl('circle',{cx:x(p.V),cy:y(p.A),r:7,fill:color,stroke:'white','stroke-width':2,class:'plot-point','data-plot-index':i,tabindex:0,role:'button','aria-label':`${p.aspect} V ${fmt(p.V)} A ${fmt(p.A)}`});point.append(svgEl('title',{},`${p.aspect}：V ${fmt(p.V)}，A ${fmt(p.A)}`));point.addEventListener('click',()=>selectPoint(i));point.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectPoint(i)}});group.append(point);const txt=svgEl('text',{x:x(p.V)+(p.V>7?-11:11),y:y(p.A)-18,'text-anchor':p.V>7?'end':'start',fill:color,'font-size':12,'font-weight':600},p.aspect.length>15?p.aspect.slice(0,13)+'…':p.aspect);group.append(txt);svg.append(group)});
 graph.append(svg);const legends=$('chart-legend');legends.replaceChildren(...result.predictions.map((p,i)=>{const b=create('button');b.type='button';b.dataset.legendIndex=i;const dot=create('i');dot.style.background=palette[i%palette.length];b.append(dot,document.createTextNode(p.aspect));b.addEventListener('click',()=>selectPoint(i));return b}));selectPoint(0)}
function renderScores(result){const grid=$('score-cards');grid.replaceChildren(...result.predictions.map((p,i)=>{const color=palette[i%palette.length];let card=create('button','new-score-card');card.type='button';card.dataset.scoreIndex=i;card.style.setProperty('--line-color',color);const heading=create('div','new-score-name');const dot=create('i');dot.style.background=color;heading.append(dot,document.createTextNode(p.aspect));card.append(heading);
 for(const [label,val] of [['V',p.V],['A',p.A]]){const line=create('div','new-score-line');line.append(create('span','',label));const track=create('div','new-score-track');const fill=create('i');fill.style.width=((clip(val)-1)/8*100)+'%';if(label==='A')fill.style.background='#5fbeb4';track.append(fill);line.append(track,create('strong','',fmt(val)));card.append(line)}
 if(result.compare){card.append(create('div','new-candidate',`实验候选 V ${fmt(p.candidate_V)} / A ${fmt(p.candidate_A)} · ΔV ${fmt(p.delta_V)} · ΔA ${fmt(p.delta_A)}`))}card.addEventListener('click',()=>selectPoint(i));return card}));const headers=['方面','当前 V','当前 A',...(result.compare?['候选 V','候选 A','ΔV','ΔA']:[])];const tr=create('tr');headers.forEach(v=>tr.append(create('th','',v)));$('score-head').replaceChildren(tr);
 $('score-body').replaceChildren(...result.predictions.map(p=>{const row=create('tr');[p.aspect,fmt(p.V),fmt(p.A),...(result.compare?[fmt(p.candidate_V),fmt(p.candidate_A),fmt(p.delta_V),fmt(p.delta_A)]:[])].forEach(v=>row.append(create('td','',v)));return row}))}
function renderPrediction(result){current=result;$('prediction-empty').hidden=true;$('prediction-results').hidden=false;$('result-time').textContent=(Number(result.elapsed_ms)/1000).toFixed(2)+' 秒 · '+result.predictions.length+' 个方面';renderTextHighlights(result.text,result.aspects);renderScatter(result);renderScores(result);
 $('candidate-notice').hidden=!result.compare;$('candidate-notice').textContent=result.compare?'历史候选与当前冻结模型为不同实验来源。本页仅作分数比较，不能推断候选模型性能更好。':'';$('warnings').textContent=(result.warnings||[]).join(' ');syncInput()}
function renderHistory(){const section=$('history-section');section.hidden=history.length===0;$('history').replaceChildren(...history.map(r=>{const b=create('button');b.type='button';b.append(create('small','',r.aspects.join(' · ')),document.createTextNode(r.text));b.addEventListener('click',()=>{$('comment').value=r.text;$('aspects').value=r.aspects.join('，');$('compare').checked=r.compare;renderPrediction(r);navigate('predict')});return b}))}
function csvCell(s){let str=String(s??'');if(/^[\s]*[=+@\-]/.test(str))str="'"+str;return '"'+str.replace(/"/g,'""')+'"'}
function saveFile(name,type,content){const url=URL.createObjectURL(new Blob([content],{type}));const a=create('a');a.download=name;a.href=url;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('文件已导出')}
function exportCSV(){if(!current)return;const rows=[['评论','方面','正式模型','V','A','候选模型','候选V','候选A','分析时间']];current.predictions.forEach(p=>rows.push([current.text,p.aspect,current.models.current.id,fmt(p.V),fmt(p.A),current.compare?current.models.candidate.id:'',current.compare?fmt(p.candidate_V):'',current.compare?fmt(p.candidate_A):'',current.created_at]));saveFile('观感-情绪分析.csv','text/csv;charset=utf-8','\ufeff'+rows.map(r=>r.map(csvCell).join(',')).join('\r\n'))}
function renderAmbiguity(){const container=$('ambiguity-chart');container.replaceChildren(...ambiguity.map(d=>{const row=create('div','error-bar'+(d.strong?' strong':'')+(d.highlight?' highlight':''));row.append(create('span','label',d.label+' · '+d.n));const track=create('div','bar-track');const bar=create('div','bar-fill');bar.style.width=(d.value/.55*100)+'%';track.append(bar);row.append(track,create('strong','',d.value.toFixed(4)));return row}))}
function renderExperiments(){const scope=document.querySelector('[data-exp-scope].active')?.dataset.expScope||'seeds',info=expGroups[scope];$('exp-chart-title').textContent=info.title;$('exp-scope-note').textContent=info.note;$('experiment-insight').textContent=info.insight;const root=$('experiment-plot');root.replaceChildren();const data=info.items;
 const w=900,h=data.length*53+96;let min=Math.min(...data.map(x=>x.value))-.00030,max=Math.max(...data.map(x=>x.value))+.0003;const plot={x:220,y:30,w:610};const x=v=>plot.x+(v-min)/(max-min)*plot.w;const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':info.title});
 for(let j=0;j<=4;j++){let v=min+(max-min)*j;svg.append(svgEl('path',{d:`M${x(v)} ${plot.y-10}V${h-50}`,stroke:'#e7e9f0'}));svg.append(svgEl('text',{x:x(v),y:h-25,'text-anchor':'middle',fill:'#8b8f9a','font-size':11},v.toFixed(4)))}
 data.forEach((d,i)=>{
   const y=plot.y+25+i*53;
   const right=x(d.value);
   const length=Math.max(1,right-plot.x);
   const group=svgEl('g',{class:'exp-v41-row',tabindex:0,role:'img',
     'aria-label':`${d.name}，综合 MAE ${d.value.toFixed(6)}`});
   const label=svgEl('text',{x:plot.x-16,y:y+5,'text-anchor':'end',
     fill:d.baseline?'#273969':'#707582','font-size':13,'font-weight':d.baseline?700:500,
     class:'exp-v41-label'},d.name);
   const line=svgEl('line',{x1:plot.x,y1:y,x2:right,y2:y,
     stroke:d.baseline?'#9698a9':'#7768e6','stroke-width':d.baseline?3:2.3,
     class:'exp-v41-line',
     'stroke-dasharray':length.toFixed(3),
     'stroke-dashoffset':length.toFixed(3),
     style:`--v41-len:${length.toFixed(3)}px;--v41-delay:${(i*.095).toFixed(3)}s`});
   const dot=svgEl('circle',{cx:right,cy:y,r:d.baseline?7:6,
     fill:d.baseline?'#24283b':'#715fe9',stroke:'white','stroke-width':2,
     class:'exp-v41-dot',style:`--v41-delay:${(i*.095+.42).toFixed(3)}s`});
   dot.append(svgEl('title',{},`${d.name}：${d.value.toFixed(6)}`));
   const score=svgEl('text',{x:Math.min(right+11,838),y:y+5,
     fill:d.baseline?'#24283b':'#715fe9','font-size':13,'font-weight':600,
     class:'exp-v41-value',style:`--v41-delay:${(i*.095+.46).toFixed(3)}s`},d.value.toFixed(6));
   group.append(label,line,dot,score);
   svg.append(group);
 });
 svg.append(svgEl('text',{x:plot.x,y:14,fill:'#8c92a0','font-size':11},'局部缩放的横轴；标出真实数值，不从 0 开始'));root.append(svg);
 const tbody=$('experiment-table');tbody.replaceChildren(...data.map(d=>{const r=create('tr');for(const s of [d.name,scope==='dev'?'官方 Dev':scope==='seeds'?'3 种子 Group CV':'Seed 42 Group CV',d.value.toFixed(6),d.baseline?'正式基线':'研究候选'])r.append(create('td','',s));return r}))}
function v5StoryVisual(index){
 if(index===0)return `<div class="v5-slide-graphic v5-graphic-space"><div class="v5-graphic-head"><span>ASPECT / EMOTION SPACE</span><b>V × A</b></div><svg viewBox="0 0 450 310" role="img" aria-label="概念示例：同一评论中的服务和价格位于不同情绪坐标"><defs><linearGradient id="v5StoryLine" x1="0" y1="1" x2="1" y2="0"><stop stop-color="#74d9d0"/><stop offset="1" stop-color="#b69afd"/></linearGradient></defs><path d="M40 35V268H419 M40 93H419 M40 151H419 M40 209H419 M134 35V268 M229 35V268 M324 35V268" stroke="#ffffff" stroke-opacity=".16" fill="none"/><path d="M188 154Q242 124 310 98" stroke="url(#v5StoryLine)" stroke-width="3" stroke-dasharray="5 8" fill="none"/><circle cx="188" cy="154" r="20" fill="#74d9d0" opacity=".22"/><circle cx="188" cy="154" r="8" fill="#74d9d0" stroke="#fff" stroke-width="3"/><circle cx="310" cy="98" r="22" fill="#b69afd" opacity=".24"/><circle cx="310" cy="98" r="9" fill="#b69afd" stroke="#fff" stroke-width="3"/><text fill="#fff" x="137" y="139" font-size="15">价格</text><text fill="#fff" x="341" y="76" font-size="15">服务</text><text fill="#abb0d1" x="40" y="23" font-size="12">A ↑</text><text fill="#abb0d1" x="419" y="297" font-size="12" text-anchor="end">V →</text></svg><div class="v5-graphic-foot"><span>一个句子</span><span>两个方面</span><span>两组坐标</span></div></div>`;
 if(index===1)return `<div class="v5-slide-graphic v5-label-study"><div class="v5-graphic-head"><span>LABEL STRUCTURE</span><b>01 / DATA</b></div><div class="v5-study-ring"><svg viewBox="0 0 220 220" role="img" aria-label="7476 个聚合方面实例，来自 8523 条观点标注"><circle cx="110" cy="110" r="83" fill="none" stroke="#ffffff20" stroke-width="18"/><circle cx="110" cy="110" r="83" fill="none" stroke="#b6a0ff" stroke-width="18" stroke-linecap="round" stroke-dasharray="457.5 521.5" transform="rotate(-90 110 110)"/></svg><div><strong>7,476</strong><span>聚合方面实例</span></div></div><div class="v5-study-cols"><div><b>8,523</b><span>原始观点标注</span></div><div><b>914</b><span>多观点方面</span></div><div><b>87</b><span>正负混合</span></div></div></div>`;
 if(index===2)return `<div class="v5-slide-graphic v5-evidence-study"><div class="v5-graphic-head"><span>VALIDATION / MODEL SELECTION</span><b>02 / PROOF</b></div><div class="v5-evidence-panel"><span>官方 Dev · 115 方面 · MAE 越低越好</span><div class="v5-evidence-row"><b>BERT L1</b><div><i style="width:83.7%"></i></div><strong>0.3067</strong></div><div class="v5-evidence-row alt"><b>Opinion-Aware</b><div><i style="width:85%"></i></div><strong>0.3115</strong></div></div><div class="v5-evidence-note">两项为同一 Dev 口径的已审核结果。<br/>改进 Group CV 并不必然改善 Dev。</div><div class="v5-graphic-foot"><span>GROUP CV / 训练内评估</span><span>DEV / 已验收</span></div></div>`;
 return `<div class="v5-slide-graphic v5-conclusion-study"><div class="v5-graphic-head"><span>MIXED SENTIMENT / VERIFIED SUBSET</span><b>04 / EVIDENCE</b></div><div class="v5-mini-eyebrow">混合情绪样本 · MAE 越低越好</div><div class="v5-comparison-row"><span>原始 BERT L1</span><strong>0.4647</strong><div class="v5-comparison-track"><i style="width:92.94%"></i></div></div><div class="v5-comparison-row improved"><span>Opinion-Aware</span><strong>0.4432</strong><div class="v5-comparison-track"><i style="width:88.64%"></i></div></div><p class="v5-evidence-caveat">仅混合子集的改善，不代表整体 Dev 超越正式模型。</p><div class="v5-validation-tile"><span>正式模型选择依据</span><strong>BERT L1 · EPOCH 3</strong><small>官方 Dev 0.3067 · 独立于上方子集</small></div></div>`;
}
// Meaningful footnotes replace the identical placeholder under all twelve metrics.
const storyMetricNotes = [
  ['效价与唤醒度','每个维度 1–9 分','针对指定方面预测'],
  ['原始文本规模','含有多条观点的方面','存在正负观点冲突'],
  ['已有实验记录','不等于独立测试次数','已有 Dev 验收结果'],
  ['混合观点子集','同一子集口径比较','正式交付模型']
];
function fillStorySlide(target,index){
 const s=storySlides[index];target.replaceChildren();
 const layout=create('div','v5-story-main'),left=create('div','v5-story-copy');
 const chapter=create('div','story-chapter',s.chapter),h=create('h2'),desc=create('p','',s.copy);
 // Reveal actual title lines separately, retaining the semantic H2 and em emphasis.
 const titleParts=s.title.match(/^([\s\S]*?)<em>([\s\S]*?)<\/em>$/);
 if(titleParts){
   [titleParts[1],titleParts[2]].forEach((text,i)=>{
     const mask=create('span','v53-title-line'),ink=create(i===1?'em':'span','v53-title-ink',text);
     mask.style.setProperty('--v53-order',String(i));mask.append(ink);h.append(mask);
   });
 }else h.innerHTML=s.title;
 left.append(chapter,h,desc);
 const graphic=create('div','v5-story-visual');graphic.innerHTML=v5StoryVisual(index);
 layout.append(left,graphic);
 const stats=create('div','story-slide-visual');
 s.stats.forEach(([num,label],n)=>{
   const item=create('div');
   item.append(create('span','',label),create('strong','',num),create('p','',storyMetricNotes[index][n]));
   stats.append(item);
 });
 target.append(layout,stats);target.dataset.slideIndex=String(index);
}
// Four slides share the tallest content height at the current viewport width.
// This is measured off-screen rather than guessed with a fixed pixel height:
// the frame never jumps when the slide changes, including on narrow screens.
let storySizing=false;
function equalizeStoryCanvas(){
 const stage=$('story-stage');
 if(storySizing||activePage!=='story'||!stage||stage.closest('[hidden]')||!stage.offsetWidth)return;
 storySizing=true;
 const measurement=create('div','story-stage v51-measure-stage');
 measurement.setAttribute('aria-hidden','true');
 const rect=stage.getBoundingClientRect();
 measurement.style.cssText=`position:fixed;left:0;top:-500vh;width:${rect.width}px;height:auto!important;min-height:0!important;max-height:none!important;display:block!important;overflow:visible!important;visibility:hidden!important;pointer-events:none!important;transition:none!important;`;
 document.body.append(measurement);
 let maxContent=0;
 for(let i=0;i<storySlides.length;i++){
   const temp=create('div','story-slide');fillStorySlide(temp,i);
   measurement.replaceChildren(temp);
   maxContent=Math.max(maxContent,Math.ceil(temp.getBoundingClientRect().height),temp.scrollHeight);
 }
 measurement.remove();
 const style=getComputedStyle(stage),pad=parseFloat(style.paddingTop)+parseFloat(style.paddingBottom);
 stage.style.setProperty('--v51-story-height',Math.ceil(Math.max(620,maxContent+pad+28))+'px');
 storySizing=false;
}
/* V5.2: Persistent, direction-aware research pager.
 * Stable dot geometry + single traveling gradient capsule. The DOM stays mounted
 * across slides, so switching left/right has real spatial continuity.
 */
let storyPagerAnim=null;
let storyPagerTarget=null;
function ensureStoryPager(){
 const dots=$('story-dots');
 if(dots.dataset.v52Ready==='1')return;
 dots.dataset.v52Ready='1';
 dots.replaceChildren();
 for(let i=0;i<storySlides.length;i++){
   const button=create('button','story-dot');
   button.type='button';
   button.setAttribute('aria-label',`跳到第 ${i+1} 页`);
   button.addEventListener('click',()=>storyGo(i));
   dots.append(button);
 }
 const capsule=create('span','story-pager-pill');
 capsule.setAttribute('aria-hidden','true');
 dots.append(capsule);
 const counter=$('story-counter');
 counter.replaceChildren(create('span','story-counter-current','01'),create('span','story-counter-total',` / ${String(storySlides.length).padStart(2,'0')}`));
}
function updateStoryPager({animate=false,direction=0}={}){
 ensureStoryPager();
 const dots=$('story-dots');
 const buttons=Array.from(dots.querySelectorAll('button'));
 buttons.forEach((button,index)=>{
   const selected=index===storyIndex;
   button.classList.toggle('active',selected);
   if(selected){button.setAttribute('aria-current','page');button.setAttribute('aria-label',`当前第 ${index+1} 页`)}
   else {button.removeAttribute('aria-current');button.setAttribute('aria-label',`跳到第 ${index+1} 页`)}
 });
 const capsule=dots.querySelector('.story-pager-pill');
 const selected=buttons[storyIndex];
 // Fixed slots: their centers stay put while the capsule travels between them.
 const target=selected.offsetLeft+(selected.offsetWidth-capsule.offsetWidth)/2;
 const motion=animate&&!reduced()&&typeof capsule.animate==='function';
 // renderStory runs during the outgoing slide animation. Do not cut short the
 // capsule's independent 490ms flight when it is already headed to this page.
 const continuing=!animate&&storyPagerAnim&&storyPagerTarget===target;
 if(!continuing){
 const matrix=getComputedStyle(capsule).transform;
 const source=matrix==='none'?Number(capsule.dataset.x||0):new DOMMatrixReadOnly(matrix).m41;
 if(storyPagerAnim){storyPagerAnim.cancel();storyPagerAnim=null;}
 capsule.style.transform=`translate3d(${source}px,0,0)`;
 storyPagerTarget=target;
 if(motion){
   const dx=target-source;
   const sign=Math.sign(dx)||direction||1;
   const animation=capsule.animate([
     {transform:`translate3d(${source}px,0,0) scaleX(1)`,offset:0},
     {transform:`translate3d(${source+dx*.36}px,0,0) scaleX(${Math.min(1.48,1.12+Math.abs(dx)*.005)})`,offset:.37},
     {transform:`translate3d(${target+sign*2.6}px,0,0) scaleX(.91)`,offset:.78},
     {transform:`translate3d(${target}px,0,0) scaleX(1)`,offset:1}
   ],{duration:490,easing:'cubic-bezier(.22,1,.36,1)',fill:'both'});
   storyPagerAnim=animation;
   animation.finished.then(()=>{
     if(storyPagerAnim===animation){
       capsule.style.transform=`translate3d(${target}px,0,0)`;
       capsule.dataset.x=String(target);
       animation.cancel();storyPagerAnim=null;
     }
   }).catch(()=>{});
 }else{
   capsule.style.transform=`translate3d(${target}px,0,0)`;
   capsule.dataset.x=String(target);
 }
 }
 const label=$('story-counter').querySelector('.story-counter-current');
 const next=String(storyIndex+1).padStart(2,'0');
 if(label.textContent!==next){
   const old=label.textContent;
   // The old numeral exits opposite to the selected slide direction; the
   // incoming numeral rolls in along that direction rather than flashing.
   const previous=$('story-counter').querySelector('.story-counter-ghost');
   if(previous)previous.remove();
   label.textContent=next;
   if(motion){
     const ghost=create('span','story-counter-ghost',old);
     label.parentElement.append(ghost);
     const dir=direction||1;
     const exit=ghost.animate([
       {transform:'translateY(0)',opacity:1},
       {transform:`translateY(${-dir*12}px)`,opacity:0}
     ],{duration:260,easing:'cubic-bezier(.4,0,1,1)',fill:'forwards'});
     const enter=label.animate([
       {transform:`translateY(${dir*12}px)`,opacity:0},
       {transform:'translateY(0)',opacity:1}
     ],{duration:370,easing:'cubic-bezier(.22,1,.36,1)'});
     exit.finished.then(()=>ghost.remove()).catch(()=>ghost.remove());
     enter.finished.catch(()=>{});
   }
 }
 $('story-progress').style.width=((storyIndex+1)/storySlides.length*100)+'%';
 $('story-prev').disabled=storyIndex===0;
 $('story-next').disabled=storyIndex===storySlides.length-1;
}

function measureTextLines(node){
 const text=(node.dataset.originalText||node.textContent||'').replace(/\s+/g,' ').trim();
 if(!text)return [];
 const probe=document.createElement('span');
 probe.style.position='absolute';probe.style.visibility='hidden';probe.style.whiteSpace='pre-wrap';probe.style.inlineSize=getComputedStyle(node).width;probe.style.font=getComputedStyle(node).font;probe.style.letterSpacing=getComputedStyle(node).letterSpacing;probe.style.lineHeight=getComputedStyle(node).lineHeight;probe.style.wordBreak=getComputedStyle(node).wordBreak;probe.textContent=text;document.body.append(probe);
 const walker=document.createTreeWalker(probe,NodeFilter.SHOW_TEXT);const tn=walker.nextNode();
 const ranges=[];let start=0,lastTop=null;
 for(let i=1;i<=tn.length;i++){
  const r=document.createRange();r.setStart(tn,start);r.setEnd(tn,i);const rects=[...r.getClientRects()].filter(x=>x.width>0&&x.height>0);
  if(!rects.length)continue;const top=Math.round(rects[rects.length-1].top);
  if(lastTop===null)lastTop=top;
  if(top!==lastTop){
   const prev=document.createRange();prev.setStart(tn,start);prev.setEnd(tn,i-1);ranges.push(tn.textContent.slice(start,i-1).trim());start=i-1;lastTop=top;
  }
 }
 if(start<tn.length)ranges.push(tn.textContent.slice(start).trim());
 probe.remove();
 return ranges.filter(Boolean);
}
function prepareOverviewCopyLines(){
 const paragraph=document.querySelector('.overview .hero-description');
 if(!paragraph||activePage!=='overview')return 0;
 const original=paragraph.dataset.originalText||(paragraph.textContent||'').replace(/\s+/g,' ').trim();
 if(!original)return 0;
 paragraph.dataset.originalText=original;
 const lines=measureTextLines(paragraph);
 // A font-ready or resize event must not destroy a running reveal when the
 // visual line breaks are unchanged. Replacing the DOM resets CSS animations.
 const oldLines=[...paragraph.querySelectorAll('.hero-copy-line .hero-copy-ink')].map(n=>n.textContent);
 if(oldLines.length===lines.length&&oldLines.every((value,i)=>value===lines[i]))return lines.length;
 paragraph.replaceChildren();
 const sr=create('span','v53-sr-only',original);paragraph.append(sr);
 lines.forEach((line,i)=>{
   const mask=create('span','hero-copy-line');mask.setAttribute('aria-hidden','true');mask.style.setProperty('--hero-delay-order',String(i));
   const ink=create('span','hero-copy-ink',line);mask.append(ink);paragraph.append(mask);
 });
 return lines.length;
}
let overviewHeroToken=0;
function playOverviewHeroIntro(){
 const hero=document.querySelector('.overview .fernway-hero');
 if(!hero)return;
 prepareOverviewCopyLines();
 hero.classList.remove('v54-hero-run','v54-hero-ready');
 if(reduced()){hero.classList.add('v54-hero-ready');return;}
 const left=[
  ...hero.querySelectorAll('.hero-navline .hero-seq-item'),
  ...hero.querySelectorAll('.fernway-copy .deck-eyebrow'),
  ...hero.querySelectorAll('.hero-title-line'),
  ...hero.querySelectorAll('.hero-description .hero-copy-line'),
  ...hero.querySelectorAll('.deck-buttons .hero-seq-item'),
  ...hero.querySelectorAll('.hero-confidence')
 ];
 left.forEach((el,i)=>el.style.setProperty('--hero-delay',`${0.04+i*0.08}s`));
 const rightOrder=[
  ['.hero-aura',0.22],['.hero-orbits',0.28],['.hero-blob',0.34],['.hero-panel-main',0.46],['.hero-panel-fact',0.58],['.hero-panel-score',0.66],['.hero-side-caption',0.76]
 ];
 rightOrder.forEach(([sel,delay])=>{const el=hero.querySelector(sel);if(el)el.style.setProperty('--hero-delay',`${delay}s`)});
 // restart CSS entrance for left copy.
 void hero.offsetWidth;
 hero.classList.add('v54-hero-run');
 ++overviewHeroToken;
 const animateLayer=(selector,keyframes,options)=>{
   const el=hero.querySelector(selector);
   if(!el||typeof el.animate!=='function')return null;
   el.getAnimations().forEach(a=>a.cancel());
   return el.animate(keyframes,{fill:'both',easing:'cubic-bezier(.2,.85,.3,1)',...options});
 };
 animateLayer('.hero-aura',[{opacity:0,transform:'scale(.9)'},{opacity:.78,transform:'scale(1)'}],{duration:900,delay:220,easing:'cubic-bezier(.17,.84,.44,1)'});
 animateLayer('.hero-orbits',[{opacity:0,transform:'scale(.95) rotate(-5deg)'},{opacity:.8,transform:'none'}],{duration:860,delay:280});
 animateLayer('.hero-blob',[{opacity:0,transform:'translateY(20px) scale(.86) rotate(-22deg)'},{opacity:1,transform:'rotate(-14deg)'}],{duration:980,delay:340,easing:'cubic-bezier(.17,.84,.44,1)'});
 animateLayer('.hero-panel-main',[{opacity:0,transform:'translateY(28px) rotate(7deg) scale(.96)'},{opacity:1,transform:'rotate(3deg) scale(1)'}],{duration:820,delay:460,easing:'cubic-bezier(.19,.87,.32,1)'});
 animateLayer('.hero-panel-fact',[{opacity:0,transform:'translateX(-16px) translateY(18px) scale(.96)'},{opacity:1,transform:'none'}],{duration:720,delay:580});
 animateLayer('.hero-panel-score',[{opacity:0,transform:'translateX(16px) translateY(18px) scale(.96)'},{opacity:1,transform:'none'}],{duration:720,delay:660});
 animateLayer('.hero-side-caption',[{opacity:0,transform:'rotate(180deg) translateY(18px)'},{opacity:.77,transform:'rotate(180deg) translateY(0)'}],{duration:620,delay:760});
 const chart=hero.querySelector('.hero-chart');
 if(chart){
  const grid=chart.querySelectorAll('g path');
  const curve=chart.querySelector('path[stroke="url(#v5Line)"]');
  const circles=[...chart.querySelectorAll('circle')];
  const labels=[...chart.querySelectorAll('g text')];
  grid.forEach((el,i)=>{el.getAnimations().forEach(a=>a.cancel());el.animate([{opacity:0},{opacity:.9}],{duration:420,delay:860+i*45,easing:'ease-out',fill:'both'})});
  if(curve){curve.getAnimations().forEach(a=>a.cancel());curve.animate([{opacity:0,strokeDashoffset:180},{opacity:.85,strokeDashoffset:0}],{duration:720,delay:1040,easing:'cubic-bezier(.2,.85,.3,1)',fill:'both'})}
  circles.forEach((el,i)=>{el.getAnimations().forEach(a=>a.cancel());el.animate([{opacity:0,transform:'scale(.6)'},{opacity:1,transform:'scale(1)'}],{duration:380,delay:1240+i*80,easing:'cubic-bezier(.17,.84,.44,1)',fill:'both'})});
  labels.forEach((el,i)=>{el.getAnimations().forEach(a=>a.cancel());el.animate([{opacity:0,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}],{duration:340,delay:1340+i*20,easing:'ease-out',fill:'both'})});
 }
}

/* V5.3 — real line-by-line typography and independently revealed research graphics.
 * Measuring glyph rectangles lets the paragraph animate along its actual visual
 * line breaks instead of fading as a single block; recalculated on resizing.
 */
function prepareStoryCopyLines(){
 const paragraph=$('story-slide')?.querySelector('.v5-story-copy>p');
 if(!paragraph||activePage!=='story')return 0;
 const original=paragraph.dataset.originalText||paragraph.textContent||'';
 paragraph.dataset.originalText=original;
 paragraph.textContent=original;
 if(!original)return 0;
 const node=paragraph.firstChild;
 const range=document.createRange(),parts=[];
 for(let i=0;i<original.length;){
   const length=original.codePointAt(i)>65535?2:1;
   range.setStart(node,i);range.setEnd(node,i+length);
   const rect=range.getBoundingClientRect();
   const top=Math.round(rect.top*2)/2;
   const last=parts[parts.length-1];
   if(last&&Math.abs(last.top-top)<2)last.text+=original.slice(i,i+length);
   else parts.push({top,text:original.slice(i,i+length)});
   i+=length;
 }
 range.detach?.();
 // If the page is temporarily unrendered, do not replace readable text.
 if(!parts.length||parts.length>24)return 0;
 paragraph.replaceChildren();
 const sr=create('span','v53-sr-only',original);
 paragraph.append(sr);
 parts.forEach((line,i)=>{
   const mask=create('span','v53-copy-line');mask.setAttribute('aria-hidden','true');
   mask.style.setProperty('--v53-order',String(i));
   const inner=create('span','v53-copy-ink',line.text);mask.append(inner);paragraph.append(mask);
 });
 return parts.length;
}
let storyRevealSerial=0;
function renderStory({reveal=activePage==='story'}={}){
 const slide=$('story-slide');
 ++storyRevealSerial;
 fillStorySlide(slide,storyIndex);
 updateStoryPager();
 equalizeStoryCanvas();
 if(activePage!=='story')return;
 prepareStoryCopyLines();
 if(reveal&&!reduced()){
   // All components mount at once but have independent keyframes and delays.
   // Animation fill-mode 'both' prevents any initial flash before the first paint.
   slide.classList.add('v53-story-enter');
 }
}
/* V4.2.1 cinematic slide transitions: click, keyboard, dots and touch share this path. */
let storyMotionToken=0;
let storyMotionAnim=null;
async function storyGo(i){
  if(!Number.isInteger(i)||i<0||i>=storySlides.length||i===storyIndex)return;
  const slide=$('story-slide');
  const dir=i>storyIndex?1:-1;
  const token=++storyMotionToken;
  if(storyMotionAnim){storyMotionAnim.cancel();storyMotionAnim=null;}
  storyIndex=i;
  // Start the indicator and counter motion immediately, synchronized with the
  // outgoing slide rather than waiting until the content has already changed.
  updateStoryPager({animate:true,direction:dir});
  // Direct render is intentional for accessibility and browsers without Web Animations.
  if(reduced()||typeof slide.animate!=='function'){
    renderStory();
    return;
  }
  const outgoing=slide.animate([
    {opacity:1,transform:'translate3d(0,0,0) scale(1)',filter:'blur(0px)'},
    {opacity:0,transform:`translate3d(${-dir*80}px,-3px,0) scale(.988)`,filter:'blur(3px)'}
  ],{duration:205,easing:'cubic-bezier(.4,0,1,1)',fill:'both'});
  storyMotionAnim=outgoing;
  try{await outgoing.finished;}catch(_){}
  if(token!==storyMotionToken)return;
  outgoing.cancel();
  renderStory();
  // New content now exists; animate the container and let CSS stagger its children.
  const incoming=slide.animate([
    {opacity:.82,transform:`translate3d(${dir*33}px,5px,0) scale(.994)`,filter:'blur(1px)'},
    {opacity:1,transform:'translate3d(0,0,0) scale(1)',filter:'blur(0px)'}
  ],{duration:400,easing:'cubic-bezier(.22,1,.36,1)',fill:'both'});
  storyMotionAnim=incoming;
  try{await incoming.finished;}catch(_){}
  if(token===storyMotionToken){incoming.cancel();storyMotionAnim=null;}
}
let touchStart=null;
function initProductPanels(){
  const activate=(buttonSelector,panelSelector,index)=>{
    document.querySelectorAll(buttonSelector).forEach((button,i)=>{
      const on=i===index;
      button.classList.toggle('active',on);
      button.setAttribute('aria-pressed',String(on));
    });
    document.querySelectorAll(panelSelector).forEach((panel,i)=>{
      const on=i===index;
      panel.hidden=!on;
      panel.classList.toggle('active',on);
      if(on&&!reduced()&&panel.animate)panel.animate([{opacity:.48,transform:'translateY(10px)'},{opacity:1,transform:'translateY(0)'}],{duration:290,easing:'cubic-bezier(.2,.85,.3,1)'});
    });
  };
  document.querySelectorAll('[data-workflow-step]').forEach((button,index)=>{
    button.addEventListener('click',()=>activate('[data-workflow-step]','[data-workflow-preview]',index));
  });
  document.querySelectorAll('[data-data-focus]').forEach((button,index)=>{
    button.addEventListener('click',()=>activate('[data-data-focus]','[data-data-panel]',index));
  });
}
function initAmbientMotion(){
  const surface=document.querySelector('.v41-gradient-feature');
  if(!surface||reduced())return;
  let frame=0;
  surface.addEventListener('pointermove',event=>{
    if(event.pointerType==='touch')return;
    const box=surface.getBoundingClientRect();
    const x=(event.clientX-box.left)/box.width-.5;
    const y=(event.clientY-box.top)/box.height-.5;
    if(frame)cancelAnimationFrame(frame);
    frame=requestAnimationFrame(()=>{
      surface.style.setProperty('--v41-mouse-x',`${(x*24).toFixed(1)}px`);
      surface.style.setProperty('--v41-mouse-y',`${(y*16).toFixed(1)}px`);
      frame=0;
    });
  },{passive:true});
  surface.addEventListener('pointerleave',()=>{
    if(frame)cancelAnimationFrame(frame);
    frame=0;
    surface.style.setProperty('--v41-mouse-x','0px');
    surface.style.setProperty('--v41-mouse-y','0px');
  });
}
function init(){
 window.addEventListener('hashchange',route);$('menu-toggle').addEventListener('click',toggleMenu);$('sidebar-backdrop').addEventListener('click',closeMenu);$('go-story').addEventListener('click',()=>navigate('story'));route();
 document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>preset(Number(b.dataset.example))));
 ['comment','aspects','compare'].forEach(id=>$(id).addEventListener('input',syncInput));$('clear').addEventListener('click',()=>{$('comment').value='';$('aspects').value='';showError('');syncInput();$('comment').focus()});
 $('analysis-form').addEventListener('submit',async e=>{e.preventDefault();if(busy||!ready||!meta)return;showError('');const data={text:$('comment').value.trim(),aspects:cleanAspects(),compare:$('compare').checked};if(!data.text)return showError('请输入一条评论。');if(!data.aspects.length)return showError('至少填写一个方面词。');if(data.aspects.length>8)return showError('一次最多分析八个方面。');const missing=data.aspects.filter(a=>a!=='NULL'&&!data.text.includes(a));if(missing.length)return showError('方面词不存在于原文：'+missing.join('、'));busy=true;updateControls();const ctrl=new AbortController(),timer=setTimeout(()=>ctrl.abort(),120000);
  try{const result=await getJSON('/api/predict',{method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':meta.token},body:JSON.stringify(data),signal:ctrl.signal});renderPrediction(result);history.unshift(result);history=history.slice(0,5);renderHistory();if(innerWidth<760)$('result-title').scrollIntoView({behavior:reduced()?'instant':'smooth'})}catch(err){showError(err.name==='AbortError'?'分析超时，请检查后端窗口。':err.message)}finally{clearTimeout(timer);busy=false;updateControls()}});
 $('clear-history').addEventListener('click',()=>{history=[];renderHistory();notify('本页历史已清空')});$('export-csv').addEventListener('click',exportCSV);$('export-json').addEventListener('click',()=>current&&saveFile('观感-情绪分析.json','application/json;charset=utf-8',JSON.stringify(current,null,2)));
 document.querySelectorAll('[data-exp-scope]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-exp-scope]').forEach(a=>{a.classList.toggle('active',a===b);a.setAttribute('aria-pressed',String(a===b))});renderExperiments()}));
 $('story-prev').addEventListener('click',()=>storyGo(storyIndex-1));$('story-next').addEventListener('click',()=>storyGo(storyIndex+1));$('story-stage').addEventListener('touchstart',e=>{touchStart=e.touches[0].clientX},{passive:true});$('story-stage').addEventListener('touchend',e=>{if(touchStart!==null){const dx=e.changedTouches[0].clientX-touchStart;if(Math.abs(dx)>65)storyGo(storyIndex+(dx<0?1:-1));touchStart=null}},{passive:true});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeMenu();return}if(activePage!=='story')return;const tag=e.target.tagName.toLowerCase();if(['input','textarea','select'].includes(tag))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();storyGo(storyIndex+1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();storyGo(storyIndex-1)}});
 initProductPanels();initAmbientMotion();renderAmbiguity();renderExperiments();if(activePage!=='story')renderStory({reveal:false});syncInput();loadMeta();poll();
}
document.addEventListener('DOMContentLoaded',init);


/* ===== GUANGAN V4.3 / REAL WEBGL + MOTION SYSTEM =====
 * Original, hand-written WebGL 1 shader; no CDN, no external SDK.
 * Palette and motion behavior based on the user-supplied VibeFather design brief.
 * Model output and experiment metrics remain untouched.
 */
(()=>{
 'use strict';
 const reduce=window.matchMedia('(prefers-reduced-motion: reduce)');
 const pointerFine=window.matchMedia('(pointer:fine)').matches;
 const visiblePage=el=>!!(el.getClientRects().length);
 const motionSwap='cubic-bezier(.22,1,.36,1)';
 const motionEnter='cubic-bezier(.16,1,.3,1)';
 const cl=(v,a,b)=>Math.max(a,Math.min(b,v));
 const palettes={base:[255,241,228],apricot:[255,158,122],lilac:[179,157,255],glacier:[143,211,255],butter:[255,224,138],seafoam:[158,236,208]};
 // 3-D interpolated value noise, implemented here rather than copying external GLSL.
 const frag=`precision highp float;
 varying vec2 vUv;
 uniform vec2 uRes;
 uniform vec2 uMouse;
 uniform float uStrength;
 uniform float uTime;
 uniform float uSeed;
 float h(vec3 p){p=fract(p*.1031);p+=dot(p,p.yxz+33.33);return fract((p.x+p.y)*p.z);}
 float n(vec3 x){vec3 a=floor(x),f=fract(x);f=f*f*(3.0-2.0*f);
 float a0=mix(h(a+vec3(0.,0.,0.)),h(a+vec3(1.,0.,0.)),f.x);
 float b0=mix(h(a+vec3(0.,1.,0.)),h(a+vec3(1.,1.,0.)),f.x);
 float a1=mix(h(a+vec3(0.,0.,1.)),h(a+vec3(1.,0.,1.)),f.x);
 float b1=mix(h(a+vec3(0.,1.,1.)),h(a+vec3(1.,1.,1.)),f.x);
 return mix(mix(a0,b0,f.y),mix(a1,b1,f.y),f.z);}
 void main(){
 float aspect=uRes.x/max(uRes.y,1.);
 vec2 p=vec2(vUv.x*aspect,vUv.y);
 vec2 m=vec2(uMouse.x*aspect,uMouse.y);
 vec2 delta=p-m;
 float swell=exp(-dot(delta,delta)*3.2)*uStrength;
 p-=delta*swell*.24;
 // Circular 40-second trajectory returns to the same field rather than jumping.
 float phase=6.28318530718*mod(uTime,40.)/40.;
 vec2 drift=.52*vec2(cos(phase),sin(phase));
 vec2 wobble=vec2(n(vec3(p*.91+drift,uSeed+3.7)),n(vec3(p*.92-drift.yx,uSeed+8.3)))-.5;
 vec2 w=p+wobble*.66;
 float n0=n(vec3(w*1.17+drift*.9,uSeed+.3));
 float n1=n(vec3(w*1.37-drift*.8+vec2(4.7,1.5),uSeed+1.1));
 float n2=n(vec3(w*.95+drift.yx*.75+vec2(1.4,5.4),uSeed+2.1));
 float n3=n(vec3(w*1.61-drift*.6+vec2(7.3,2.2),uSeed+5.2));
 float n4=n(vec3(w*1.22+drift*.6+vec2(3.1,9.5),uSeed+7.7));
 vec3 col=vec3(1.,.945,.894);
 col=mix(col,vec3(1.,.62,.478),smoothstep(.31,.78,n0)*.91);
 col=mix(col,vec3(.702,.616,1.),smoothstep(.24,.83,n1)*.83);
 col=mix(col,vec3(.561,.827,1.),smoothstep(.27,.84,n2)*.88);
 col=mix(col,vec3(1.,.878,.541),smoothstep(.42,.87,n3)*.70);
 col=mix(col,vec3(.62,.925,.816),smoothstep(.43,.9,n4)*.64);
 col+=vec3(swell*.067);
 col=mix(col,vec3(1.),.16*(1.-smoothstep(0.,.60,vUv.x)));
 float grain=fract(sin(dot(gl_FragCoord.xy+fract(uTime)*61.,vec2(12.9898,78.233)))*43758.5453);
 col+=(grain-.5)*.012;
 gl_FragColor=vec4(clamp(col,0.,1.),1.);
 }`;
 const vert='attribute vec2 aPos;varying vec2 vUv;void main(){vUv=(aPos+1.)*.5;gl_Position=vec4(aPos,0.,1.);}';
 function createShader(gl,type,source){
  const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error('Shader compile: '+gl.getShaderInfoLog(s));return s;
 }
 function startMesh(parent,seed){
  parent.classList.add('v43-mesh-container');
  if(reduce.matches){parent.dataset.glState='reduced';return ()=>{};}
  const canvas=document.createElement('canvas');canvas.className='v43-mesh-canvas';canvas.setAttribute('aria-hidden','true');
  let gl,program,buffer,observer,resize,raf=0,active=false,mouse=[.7,.6],target=[.7,.6],strength=0,targetStrength=0;
  let disposed=false;
  try{
   gl=canvas.getContext('webgl',{alpha:false,antialias:false,depth:false,stencil:false,preserveDrawingBuffer:false,powerPreference:'low-power'});
   if(!gl)throw Error('WebGL unavailable');
   program=gl.createProgram();gl.attachShader(program,createShader(gl,gl.VERTEX_SHADER,vert));gl.attachShader(program,createShader(gl,gl.FRAGMENT_SHADER,frag));gl.linkProgram(program);
   if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error('Program link');
   buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
   gl.useProgram(program);const attrib=gl.getAttribLocation(program,'aPos');gl.enableVertexAttribArray(attrib);gl.vertexAttribPointer(attrib,2,gl.FLOAT,false,0,0);
   parent.appendChild(canvas);
   const uniforms={time:gl.getUniformLocation(program,'uTime'),res:gl.getUniformLocation(program,'uRes'),mouse:gl.getUniformLocation(program,'uMouse'),strength:gl.getUniformLocation(program,'uStrength'),seed:gl.getUniformLocation(program,'uSeed')};
   const fit=()=>{if(disposed)return;const dpr=Math.min(devicePixelRatio||1,innerWidth<768?1:1.5),r=parent.getBoundingClientRect();
    if(!r.width||!r.height)return;const w=Math.max(2,Math.round(r.width*dpr)),h=Math.max(2,Math.round(r.height*dpr));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);}
   };
   const shouldRun=()=>!disposed&&!reduce.matches&&active&&!document.hidden;
   const stop=()=>{if(raf)cancelAnimationFrame(raf);raf=0;};
   const begin=()=>{if(!shouldRun()||raf)return;raf=requestAnimationFrame(draw)};
   function draw(now){raf=0;if(!shouldRun())return;fit();
    mouse[0]+=(target[0]-mouse[0])*.045;mouse[1]+=(target[1]-mouse[1])*.045;
    strength+=(targetStrength-strength)*.045;
    if(!pointerFine){const t=now*.001;target=[.5+.28*Math.sin(t*.21),.55+.2*Math.sin(t*.17)];targetStrength=.52;}
    gl.useProgram(program);gl.uniform1f(uniforms.time,now*.001);gl.uniform2f(uniforms.res,canvas.width,canvas.height);
    gl.uniform2f(uniforms.mouse,mouse[0],mouse[1]);gl.uniform1f(uniforms.strength,strength);gl.uniform1f(uniforms.seed,seed);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    if(!parent.classList.contains('v43-webgl-ready')){parent.classList.add('v43-webgl-ready');parent.dataset.glState='active';}
    begin();
   }
   function pointer(e){const r=parent.getBoundingClientRect();const x=(e.clientX-r.left)/r.width,y=1-(e.clientY-r.top)/r.height;
    target=[cl(x,0,1),cl(y,0,1)];targetStrength=x>=0&&x<=1&&y>=0&&y<=1?1:0;}
   function leave(){targetStrength=0;}
   function visibility(){if(document.hidden)stop();else begin();}
   const fine=pointerFine;
   if(fine){parent.addEventListener('pointermove',pointer,{passive:true});parent.addEventListener('pointerleave',leave,{passive:true});}
   observer=new IntersectionObserver(entries=>{active=entries[0].isIntersecting&&visiblePage(parent);if(active)begin();else stop();},{threshold:.04});observer.observe(parent);
   resize=new ResizeObserver(fit);resize.observe(parent);document.addEventListener('visibilitychange',visibility);
   const onReduce=()=>{if(reduce.matches){stop();canvas.style.display='none';parent.dataset.glState='reduced';}else{canvas.style.display='';parent.dataset.glState='active';begin();}};
   reduce.addEventListener?.('change',onReduce);
   fit();
   return ()=>{disposed=true;stop();observer.disconnect();resize.disconnect();document.removeEventListener('visibilitychange',visibility);
    if(fine){parent.removeEventListener('pointermove',pointer);parent.removeEventListener('pointerleave',leave);}
    reduce.removeEventListener?.('change',onReduce);
    if(gl){if(buffer)gl.deleteBuffer(buffer);if(program)gl.deleteProgram(program);gl.getExtension('WEBGL_lose_context')?.loseContext();}canvas.remove();};
  }catch(error){console.info('Guangan WebGL falls back to CSS:',error.message);parent.dataset.glState='fallback';canvas.remove();
   if(gl)try{gl.getExtension('WEBGL_lose_context')?.loseContext();}catch(_){ }return ()=>{};
  }
 }
 function initMesh(){
  const targets=[['.fernway-hero',.13],['.v41-gradient-feature',7],['.v42-predict-scene',2.7]];
  const disposers=targets.map(([selector,seed])=>document.querySelector(selector)).filter(Boolean).map((el,i)=>startMesh(el,targets.find(t=>el.matches(t[0]))[1]));
  window.addEventListener('pagehide',()=>disposers.forEach(fn=>fn()),{once:true});
 }
 function initFloat(){
  const hero=document.querySelector('.fernway-hero'),art=document.querySelector('.fernway-art');
  if(!hero||!art||reduce.matches||!pointerFine)return;
  let mx=0,my=0,tx=0,ty=0,inside=false,visible=false,frame=0;
  const depth=[.35,.75,1.2];
  const panels=[...art.querySelectorAll('.float-panel')];
  const onMove=e=>{const r=hero.getBoundingClientRect();tx=cl((e.clientX-r.left)/r.width-.5,-.5,.5);ty=cl((e.clientY-r.top)/r.height-.5,-.5,.5);inside=true;};
  const leave=()=>{inside=false;tx=0;ty=0;};
  hero.addEventListener('pointermove',onMove,{passive:true});hero.addEventListener('pointerleave',leave);
  function tick(){frame=0;if(!visible||document.hidden||reduce.matches)return;
   mx+=(tx-mx)*.075;my+=(ty-my)*.075;
   art.style.transform=`perspective(1600px) rotateY(${(-mx*6).toFixed(2)}deg) rotateX(${(my*4).toFixed(2)}deg)`;
   const y=hero.getBoundingClientRect().top;const pct=cl(-y/Math.max(hero.clientHeight,1),0,1);
   panels.forEach((p,i)=>p.style.setProperty('--v43-parallax',(-pct*38*depth[i%3]).toFixed(1)+'px'));
   frame=requestAnimationFrame(tick);
  }
  const io=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible){cancelAnimationFrame(frame);frame=0;}else if(!frame)frame=requestAnimationFrame(tick);},{threshold:.05});io.observe(hero);
  const stop=()=>{cancelAnimationFrame(frame);frame=0;};const vis=()=>{if(document.hidden)stop();else if(visible)frame=requestAnimationFrame(tick);};
  document.addEventListener('visibilitychange',vis);
  window.addEventListener('pagehide',()=>{stop();io.disconnect();document.removeEventListener('visibilitychange',vis);hero.removeEventListener('pointermove',onMove);hero.removeEventListener('pointerleave',leave);});
 }
 function initReveal(){
  if(reduce.matches||!('IntersectionObserver'in window))return;
  // Section headers and their individual text lines lead, followed by charts and content cards.
  // 20% intersection, one time only, 0.06s staggering (template-style reveal).
  const selector=[
   '.section-caption>span','.section-caption>p',
   '.fernway-copy .deck-eyebrow','.fernway-copy h1','.fernway-copy .deck-description','.fernway-copy .deck-buttons',
   '.workflow-heading h2','.workflow-heading>p','.workflow-grid',
   '.page-heading>.kicker','.page-heading h1','.page-heading>p',
   '.story-top h1','.story-help','.data-masthead','.data-focus-nav',
   '.data-layout .chart-header h2','.data-layout .content-card',
   '.exp-main .chart-header h2','.exp-main','.exp-summary-card',
   '.about-flow','.about-grid','.api-copy h2','.api-copy>p','.api-showcase',
   '.insight-card','.v44-feature-grid>div',
   '.v41-feature-content h2','.v41-feature-content p'
  ].join(',');
  const items=[...document.querySelectorAll(selector)];
  const obs=new IntersectionObserver(entries=>entries.forEach(entry=>{
   if(entry.isIntersecting){entry.target.classList.add('v43-revealed');obs.unobserve(entry.target);}
  }),{threshold:.2});
  const groups=new Map();
  items.forEach(el=>{
   const section=el.closest('.section-caption,.page-heading,.workflow-heading,.fernway-copy,.about-grid,.data-layout,.exp-visual-grid,.api-showcase,.insight-grid,.v41-gradient-feature')||el.parentElement;
   const index=groups.get(section)||0;groups.set(section,index+1);
   el.classList.add('v43-reveal');el.style.setProperty('--v43-stagger',Math.min(index,6)*.06+'s');
   obs.observe(el);
  });
  window.addEventListener('pagehide',()=>obs.disconnect(),{once:true});
 }
 function initCounting(){
  if(reduce.matches||!('IntersectionObserver'in window))return;
  const els=[...document.querySelectorAll('.insight-number,.data-masthead strong,.donut strong,.feature-metrics strong,.exp-stat-top strong')];
  const io=new IntersectionObserver(entries=>entries.forEach(entry=>{
   if(!entry.isIntersecting)return;io.unobserve(entry.target);
   const el=entry.target,original=el.textContent.trim(),commas=original.includes(','),parts=original.replaceAll(',','').split('.'),dec=parts[1]?.length||0,value=Number(original.replaceAll(',',''));
   if(!Number.isFinite(value))return;const start=performance.now();
   const animate=now=>{const t=cl((now-start)/1400,0,1),ease=1-Math.pow(1-t,3);let formatted=(value*ease).toFixed(dec);if(commas)formatted=Number(formatted).toLocaleString('en-US',{minimumFractionDigits:dec,maximumFractionDigits:dec});el.textContent=formatted;if(t<1)requestAnimationFrame(animate);else el.textContent=original;};
   requestAnimationFrame(animate);
  }),{threshold:.5});els.forEach(el=>io.observe(el));
 }
 function initAutoWorkflow(){
  const wrap=document.querySelector('.workflow-showcase');if(!wrap||reduce.matches)return;
  const buttons=[...wrap.querySelectorAll('[data-workflow-step]')];if(buttons.length<2)return;
  let active=true,hover=false,focus=false,last=performance.now(),pauseUntil=0,visible=false,frame=0;
  const period=7000;
  const step=now=>{frame=0;if(document.hidden||!visible||reduce.matches)return;
   if(hover||focus||now<pauseUntil){last=now;frame=requestAnimationFrame(step);return;}
   let elapsed=Math.max(0,now-last),portion=cl(elapsed/period,0,1);
   const current=buttons.find(b=>b.classList.contains('active'))||buttons[0];current.style.setProperty('--v43-tab-progress',`${(portion*100).toFixed(2)}%`);
   if(portion>=1){const index=buttons.indexOf(current);buttons[(index+1)%buttons.length].click();last=now;buttons.forEach(b=>b.style.setProperty('--v43-tab-progress','0%'));}
   frame=requestAnimationFrame(step);
  };
  wrap.addEventListener('pointerenter',()=>{hover=true});wrap.addEventListener('pointerleave',()=>{hover=false;last=performance.now()});
  wrap.addEventListener('focusin',()=>{focus=true});wrap.addEventListener('focusout',()=>{focus=wrap.contains(document.activeElement);last=performance.now()});
  buttons.forEach(b=>b.addEventListener('click',e=>{if(e.isTrusted)pauseUntil=performance.now()+15000;last=performance.now();buttons.forEach(btn=>btn.style.setProperty('--v43-tab-progress','0%'));}));
  const io=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting&&visiblePage(wrap);if(!frame&&visible){last=performance.now();frame=requestAnimationFrame(step);}else if(!visible){cancelAnimationFrame(frame);frame=0;}},{threshold:.15});io.observe(wrap);
  window.addEventListener('pagehide',()=>{cancelAnimationFrame(frame);io.disconnect();},{once:true});
  // Keyboard navigation mirrors the original template's vertical feature tabs.
  buttons.forEach((button,i)=>button.addEventListener('keydown',e=>{if(['ArrowDown','ArrowRight','ArrowUp','ArrowLeft'].includes(e.key)){e.preventDefault();let index=i+(['ArrowDown','ArrowRight'].includes(e.key)?1:-1);index=(index+buttons.length)%buttons.length;buttons[index].click();buttons[index].focus();}}));
 }
 function initPanelSwap(){
  // Reuse the existing real app's panel logic, add an outgoing visual clone for continuity.
  const groups=[['[data-workflow-step]','[data-workflow-preview]'],['[data-data-focus]','[data-data-panel]']];
  if(reduce.matches)return;
  for(const [control,panel] of groups){
   document.querySelectorAll(control).forEach(btn=>btn.addEventListener('click',()=>{
    const current=document.querySelector(`${panel}.active`);if(!current||!current.animate)return;
    current.animate([{opacity:.68,transform:'translateX(30px) rotateY(-4deg)'},{opacity:1,transform:'translateX(0) rotateY(0deg)'}],{duration:480,easing:motionSwap});
    const rows=current.querySelectorAll('.mock-results>div,.mock-aspect-pair>div,.focus-visual>div,.focus-side>*');
    rows.forEach((r,i)=>r.animate([{opacity:0,transform:'translateY(12px)'},{opacity:1,transform:'translateY(0)'}],{duration:420,delay:i*45,easing:motionEnter,fill:'both'}));
   }));
  }
 }
 function initExperimentMotion(){
  const holder=document.getElementById('experiment-plot');if(!holder||reduce.matches||!('MutationObserver'in window))return;
  let wait=0;const reveal=()=>{
   if(!visiblePage(holder))return;
   holder.querySelectorAll('.exp-v41-row').forEach((row,i)=>{
    // SVG CSS drawing animation already installed in V4.1; observer restarts it on every data selection.
    row.style.animation='none';row.getBoundingClientRect();row.style.animation='';
    row.style.setProperty('--v43-index',String(i));
   });
  };
  const mo=new MutationObserver(()=>{clearTimeout(wait);wait=setTimeout(reveal,35)});mo.observe(holder,{childList:true});
  window.addEventListener('pagehide',()=>{mo.disconnect();clearTimeout(wait);},{once:true});
 }
 function initMotion(){
  document.documentElement.classList.add('v43-motion');
  initMesh();initFloat();initReveal();initCounting();initAutoWorkflow();initPanelSwap();initExperimentMotion();
  if(reduce.matches)document.documentElement.classList.add('v43-reduced');
  reduce.addEventListener?.('change',e=>document.documentElement.classList.toggle('v43-reduced',e.matches));
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initMotion,{once:true});else initMotion();
})();
/* ===== END GUANGAN V4.3 ===== */

/* ===== GUANGAN V4.4 / PREMIUM HEADER + SCROLL LITERACY =====
 * Template-inspired, independently authored. No changes to API calls or numbers.
 */
(()=>{
 'use strict';
 const header=document.querySelector('.topbar');
 function updateHeader(){if(header)header.classList.toggle('v44-scrolled',window.scrollY>24)}
 window.addEventListener('scroll',updateHeader,{passive:true});updateHeader();
 const copy=document.getElementById('v44-copy-api');
 if(copy)copy.addEventListener('click',async()=>{
  const content=JSON.stringify({text:'服务很好，但价格偏贵。',aspects:['服务','价格'],compare:false},null,2);
  try{await navigator.clipboard.writeText(content);copy.textContent='已复制';}
  catch(error){copy.textContent='请手动复制';}
  window.setTimeout(()=>{copy.textContent='复制代码'},1800);
 });
})();
/* ===== END GUANGAN V4.4 ===== */


/* GUANGAN V5 — additional interactions for the verified local API showcase.
   No backend changes, no external calls, no invented inference results. */
(()=>{
 'use strict';
 const examples={
  json:{name:'request.json',source:JSON.stringify({text:'服务很好，但价格有点贵。',aspects:['服务','价格'],compare:false},null,2)},
  curl:{name:'terminal.sh',source:`# TOKEN 请先从 GET /api/meta 读取并替换
curl -X POST 'http://127.0.0.1:8877/api/predict' \\
  -H 'Content-Type: application/json' \\
  -H 'X-Demo-Token: <TOKEN>' \\
  -d '{"text":"服务很好，但价格有点贵。","aspects":["服务","价格"],"compare":false}'`},
  python:{name:'client.py',source:`import json
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8877"
with urlopen(base + "/api/meta") as response:
    token = json.load(response)["token"]
data = {"text": "服务很好，但价格有点贵。",
        "aspects": ["服务", "价格"], "compare": False}
req = Request(base + "/api/predict",
    data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json", "X-Demo-Token": token},
    method="POST")
with urlopen(req) as response:
    print(response.read().decode("utf-8"))`}
 };
 const escape=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const highlight=(source)=>{
  const re=/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b(?:true|false|null|True|False|None|import|from|with|as|curl|POST|GET)\b)/g;
  let result='',last=0,m;
  while((m=re.exec(source))){result+=escape(source.slice(last,m.index));const token=m[0];const key=token.startsWith('"')&&/^\s*:/.test(source.slice(re.lastIndex)) ? 'v5-token-key' : m[1]?'v5-token-string':'v5-token-keyword';result+=`<span class="${key}">${escape(token)}</span>`;last=re.lastIndex;}
  return result+escape(source.slice(last));
 };
 function setup(){
  const buttons=[...document.querySelectorAll('[data-v5-code]')];
  const output=document.getElementById('v5-code-content'),title=document.getElementById('v5-code-title'),copy=document.getElementById('v5-copy');
  if(!output||!title||!copy||!buttons.length)return;
  let active='json';
  const activate=key=>{
   if(!examples[key])return;active=key;
   title.textContent=examples[key].name;
   output.innerHTML=highlight(examples[key].source);
   output.parentElement.scrollTop=0;
   buttons.forEach(b=>{const selected=b.dataset.v5Code===key;b.classList.toggle('active',selected);b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;});
  };
  buttons.forEach((button,index)=>{
   button.addEventListener('click',()=>activate(button.dataset.v5Code));
   button.addEventListener('keydown',e=>{
    if(e.key==='ArrowRight'||e.key==='ArrowLeft'){
     e.preventDefault();const next=buttons[(index+(e.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length];activate(next.dataset.v5Code);next.focus();
    }
   });
  });
  copy.addEventListener('click',async()=>{
   const value=examples[active].source;
   try{
    if(!navigator.clipboard?.writeText)throw Error('clipboard unavailable');
    await navigator.clipboard.writeText(value);
    copy.textContent='已复制 ✓';
   }catch(_){
    // Local fallback also works in older browsers on 127.0.0.1.
    const area=document.createElement('textarea');area.value=value;area.setAttribute('readonly','');area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();
    const ok=typeof document.execCommand==='function'&&document.execCommand('copy');area.remove();copy.textContent=ok?'已复制 ✓':'请选中代码复制';
   }
   window.setTimeout(()=>{copy.innerHTML='复制 <span aria-hidden="true">↗</span>';},1800);
  });
  activate('json');
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',setup,{once:true});else setup();
})();

/* GUANGAN V5.1 — resize-safe equal research frames */
let v51ResizePending=false;
window.addEventListener('resize',()=>{
 if(v51ResizePending)return;
 const needsStory=activePage==='story';
 const needsOverview=activePage==='overview';
 if(!needsStory&&!needsOverview)return;
 v51ResizePending=true;
 requestAnimationFrame(()=>{
   v51ResizePending=false;
   if(activePage==='story'){equalizeStoryCanvas();prepareStoryCopyLines()}
   if(activePage==='overview'){prepareOverviewCopyLines()}
 });
},{passive:true});
if(document.fonts&&document.fonts.ready){document.fonts.ready.then(()=>{
 if(activePage==='story'){equalizeStoryCanvas();prepareStoryCopyLines()}
 if(activePage==='overview'){prepareOverviewCopyLines()}
});}
