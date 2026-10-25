// Optional Node.js smoke for the browser script with a mocked DOM/canvas.
// This checks data/event wiring, not browser layout or visual fidelity.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'../src/live_risk_dashboard.html'),'utf8');
const sample=JSON.parse(fs.readFileSync(process.argv[2]||(()=>{throw Error('usage: node live_risk_render.cjs SNAPSHOT.json')})(),'utf8'));
class Element {
 constructor(id){this.id=id;this.style={};this._value=['expiry-filter','side-filter'].includes(id)?'all':'';this.options=[];this.classList={add(){},remove(){}};this.dataset={};}
 set innerHTML(v){this._html=v;let choices=[...v.matchAll(/<option value="([^"]*)"/g)].map(m=>({value:m[1]}));if(choices.length){this.options=choices;this._value=choices[0].value;}}
 get innerHTML(){return this._html||'';}set value(v){this._value=String(v);}get value(){return this._value;}
 getBoundingClientRect(){return {width:700,height:300,left:0,top:0};}
 getContext(){return new Proxy({}, {get:(o,k)=>k in o?o[k]:(()=>{})});}
 querySelectorAll(){return [...this.innerHTML.matchAll(/<tr data-id="([^"]+)"/g)].map(m=>{let e=new Element('tr');e.dataset.id=m[1];return e;});}
 setPointerCapture(){}
}
const elements=new Map();const el=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};let eventStream;
const context={document:{getElementById:el},window:{addEventListener(){}},devicePixelRatio:1,setInterval(){},console,
 fetch:()=>Promise.resolve({json:()=>Promise.resolve(sample)}),EventSource:class {constructor(){eventStream=this;this.listeners={};}addEventListener(k,f){this.listeners[k]=f;}}};
vm.createContext(context);vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1],context,{timeout:10000});
(async()=>{await new Promise(setImmediate);assert(el('chain').innerHTML.includes('<tr data-id='));assert(el('metrics').innerHTML.includes('Vega'));assert(vm.runInContext('points.length',context)>0);
 el('heat').onclick();assert(vm.runInContext('heatmap',context));el('mesh').onclick();assert(!vm.runInContext('heatmap',context));
 el('side-filter').value='put';el('side-filter').onchange();assert(!el('chain').innerHTML.includes('<td>call</td>'));el('side-filter').value='all';el('side-filter').onchange();
 const old=vm.runInContext('snapshot.generation',context);el('pause').onclick();eventStream.listeners.snapshot({data:JSON.stringify({...sample,generation:old+1})});assert.equal(vm.runInContext('snapshot.generation',context),old);el('pause').onclick();assert.equal(vm.runInContext('snapshot.generation',context),old+1);
 let p=vm.runInContext('points[0]',context);el('surface').onpointermove({clientX:p.x,clientY:p.y});assert.equal(el('tip').style.display,'block');
 const out={status:'pass',checks:['snapshot renderer','chain and portfolio data','surface projection and heatmap','smile plot','side filter','freeze/resume SSE','surface inspection tooltip'],quote_count:sample.rows.length,surface_points:sample.grid.length,scope:'mocked DOM JavaScript runtime; browser visual review is separate'};
 console.log(JSON.stringify(out));
})().catch(e=>{console.error(e);process.exitCode=1;});
