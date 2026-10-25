#!/usr/bin/env python3
"""Bounded synthetic market replay through the actual native Volt HTTP/SSE service."""
import argparse, hashlib, json, math, os, pathlib, shutil, subprocess, threading, time, urllib.request, urllib.error
BASE=pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BINARY=BASE/'target/release/live_option_risk'

def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,v):
 t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def option_price(s,k,t,sigma,side,r=.03,q=.01):
 d1=(math.log(s/k)+(r-q+.5*sigma*sigma)*t)/(sigma*math.sqrt(t));d2=d1-sigma*math.sqrt(t)
 N=lambda x:.5*math.erfc(-x/math.sqrt(2));disc=math.exp(-q*t);c=s*disc*N(d1)-k*math.exp(-r*t)*N(d2)
 return c if side=='call' else c-s*disc+k*math.exp(-r*t)
def price(s,k,t,sigma,side):
 d1=(math.log(s/k)+(.03-.01+.5*sigma*sigma)*t)/(sigma*math.sqrt(t));n=.5*math.erfc(-d1/math.sqrt(2))
 return option_price(s,k,t,sigma,side),math.exp(-.01*t)*(n-(side=='put'))
def finite_greeks(s,k,t,sigma,side):
 # Independent central price bumps; output uses the native display's financial units.
 h=s*1e-4;dv=1e-4;dt=1/(365*1000);dr=1e-5;v=option_price(s,k,t,sigma,side)
 up=option_price(s+h,k,t,sigma,side);down=option_price(s-h,k,t,sigma,side)
 return dict(delta=(up-down)/(2*h),gamma=(up-2*v+down)/(h*h),
  vega_per_vol_point=(option_price(s,k,t,sigma+dv,side)-option_price(s,k,t,sigma-dv,side))/(2*dv)*.01,
  theta_per_day=-(option_price(s,k,t+dt,sigma,side)-option_price(s,k,t-dt,sigma,side))/(2*dt)/365,
  rho_per_bp=(option_price(s,k,t,sigma,side,r=.03+dr)-option_price(s,k,t,sigma,side,r=.03-dr))/(2*dr)*.0001)
def frame(step,start,interval=.4):
 elapsed=step*interval;cutoff=start+round(elapsed*1_000_000_000);s=100+1.3*math.sin(step*.17);out=[];truth={}
 for e,days in enumerate([30,60,90]):
  t=days/365-elapsed/(365*86400)
  for k in [90.,92.5,95.,97.5,100.,102.5,105.,107.5,110.]:
   x=math.log(k/s)-.02*t;sig=.2-.12*x+.1*x*x+.006*math.sin(step*.12)+.003*t
   for side in ['call','put']:
    key=f'REPLAY-E{e}-{side}-{k:g}';pv,d=price(s,k,t,sig,side);width=max(.003,pv*.002)
    out.append(dict(quote_id=f'{key}-{step+1}',contract_id=key,source_id='deterministic-european-replay',underlying_id='SIMX',timestamp_ns=cutoff,input=dict(spot=s,strike=k,maturity_years=t,rate=.03,dividend_yield=.01,side=side),bid=pv-width,ask=pv+width));truth[key]=(pv,d,sig)
 return dict(schema='volt.compute.quote_frame.v1',generation=step+1,cutoff_ns=cutoff,currency='USD',position_revision=1,synthetic=True,quotes=out,positions=[dict(contract_id=f'REPLAY-E1-{side}-100',quantity=-10,multiplier=100) for side in ['call','put']]),truth

def http(url,value=None):
 data=None if value is None else json.dumps(value,separators=(',',':')).encode()
 req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=5) as r:
  b=r.read(4*1024*1024+1);assert len(b)<=4*1024*1024
  return json.loads(b),sha(b)
def check(v,f,truth):
 assert v['generation']==f['generation'] and v['synthetic'] and v['spot']==f['quotes'][0]['input']['spot']
 assert v['portfolio']['complete'] and v['surface'] is not None and v['surface_error'] is None
 assert len(v['rows'])==54 and len(v['grid'])>0
 fields=['price','delta','gamma','vega_per_vol_point','theta_per_day','rho_per_bp']
 maxima={k:0. for k in fields};totals={k:0. for k in fields};finite_errors={k:0. for k in fields if k!='price'}
 for row in v['rows']:
  q=row['quote'];key=q['quote']['contract_id'];assert q['eligible'] and row['greeks'] is not None
  pv,d,sig=truth[key];g=row['greeks'];inp=q['quote']['input']
  maxima['price']=max(maxima['price'],abs(g['price']-pv));maxima['delta']=max(maxima['delta'],abs(g['delta']-d))
  fg=finite_greeks(inp['spot'],inp['strike'],inp['maturity_years'],sig,inp['side'])
  for k,val in fg.items():
   err=abs(g[k]-val);finite_errors[k]=max(finite_errors[k],err)
   assert err<2e-6+2e-4*abs(val),(key,k,g[k],val,err)
  for k in fields:totals[k]+=row['position_units']*g[k]
 assert maxima['price']<1e-6 and maxima['delta']<1e-7,maxima
 for k in fields:
  pk='value' if k=='price' else k
  assert math.isfinite(v['portfolio'][pk]) and abs(totals[k]-v['portfolio'][pk])<1e-8,(pk,totals[k],v['portfolio'][pk])
 scenario_error=0.;local_error=0.
 for scenario in v['scenarios']:
  exact=0.
  for row in v['rows']:
   if not row['position_units']:continue
   inp=row['quote']['quote']['input'];sig=truth[row['quote']['quote']['contract_id']][2]
   p1=option_price(inp['spot']*(1+scenario['spot_change']),inp['strike'],inp['maturity_years'],sig+scenario['volatility_change'],inp['side'])
   exact+=row['position_units']*(p1-truth[row['quote']['quote']['contract_id']][0])
  ds=v['spot']*scenario['spot_change'];dv=scenario['volatility_change']*100
  approx=totals['delta']*ds+.5*totals['gamma']*ds*ds+totals['vega_per_vol_point']*dv
  scenario_error=max(scenario_error,abs(exact-scenario['pnl']));local_error=max(local_error,abs(approx-scenario['local_pnl']))
 assert scenario_error<1e-5 and local_error<1e-8,(scenario_error,local_error)
 return dict(max_price_error=maxima['price'],max_delta_error=maxima['delta'],finite_difference_greek_errors=finite_errors,
  max_scenario_error=scenario_error,max_local_approximation_error=local_error,source_not_after_calculation=f['cutoff_ns']<=v['calculated_at_ns'],
  observation_age_at_cutoff_seconds=(f['cutoff_ns']-f['quotes'][0]['timestamp_ns'])/1e9,
  portfolio=v['portfolio'],spot=v['spot'],generation=v['generation'],cube_generation=v['cube']['generation'],input_sha256=v['input_sha256'],positions_sha256=v['positions_sha256'],calculation_ms=v['timings']['total_ms'])

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--binary',type=pathlib.Path,default=DEFAULT_BINARY);ap.add_argument('--port',type=int,default=57927);ap.add_argument('--frames',type=int,default=64);ap.add_argument('--interval',type=float,default=.4);ap.add_argument('--start-ns',type=int,default=None);ap.add_argument('--source-commit',default='local-build');ap.add_argument('--record',action='store_true');ap.add_argument('--linger-seconds',type=float,default=0);a=ap.parse_args()
 assert 3<=a.frames<=120 and .25<=a.interval<=5 and 0<=a.linger_seconds<=300 and 1024<a.port<65536
 a.output.mkdir(parents=True,exist_ok=False);out=a.output;start=a.start_ns if a.start_ns is not None else time.time_ns();url=f'http://127.0.0.1:{a.port}'
 binary=out/'live_option_risk';shutil.copy2(a.binary,binary);binary.chmod(0o555)
 f,_=frame(0,start,a.interval);write(out/'initial-frame.json',f)
 recipe=dict(schema='volt.demo.live_service_replay.v1',source='deterministic synthetic European options and common stock marks',analytics='actual Volt CPUf64 IV/surface/Greeks/scenarios and Hypercube generations',scope='single underlying/options-only book; 54 quotes,3 expiries,2 signed option positions; no vendor feed or cross-sectional factor service',start_ns=start,frames=a.frames,interval_seconds=a.interval,port=a.port,binary_sha256=sha(binary.read_bytes()),harness_sha256=sha(pathlib.Path(__file__).read_bytes()),source_clock='one replay second per wall second at configured interval; no accelerated future source timestamps',independent_controls='Python scalar prices/deltas; independent finite price bumps for five Greeks; all signed portfolio sums; exact full scenario repricing and local approximations',source_commit=a.source_commit,command=[str(x) for x in __import__('sys').argv])
 write(out/'protocol.json',recipe)
 proc=None;record=None;receipts=[];events=[];stream_errors=[];stop=threading.Event()
 try:
  log=open(out/'server.log','wb');proc=subprocess.Popen([str(binary),'--address',f'127.0.0.1:{a.port}','--input',str(out/'initial-frame.json'),'--serve'],stdout=log,stderr=subprocess.STDOUT)
  for _ in range(100):
   if proc.poll() is not None:raise RuntimeError('native server exited; inspect server.log')
   try:v,h=http(url+'/api/snapshot');break
   except (urllib.error.URLError,ConnectionError):time.sleep(.05)
  else:raise RuntimeError('server readiness deadline exceeded')
  receipts.append(dict(check(v,f,frame(0,start,a.interval)[1]),snapshot_sha256=h));write(out/'first-snapshot.json',v)
  def sse():
   try:
    with urllib.request.urlopen(url+'/api/stream',timeout=5) as r:
     event={}
     while not stop.is_set():
      b=r.readline(4*1024*1024+1);assert len(b)<=4*1024*1024
      if not b:break
      if b.startswith(b'id:'):event['id']=int(b[3:])
      if b.startswith(b'data:'):
       x=json.loads(b[5:]);assert x['generation']==event['id'];events.append(dict(generation=x['generation'],input_sha256=x['input_sha256']))
   except Exception as e:
    if not stop.is_set():stream_errors.append(repr(e))
  th=threading.Thread(target=sse,daemon=True);th.start()
  if a.record:
   recorder=pathlib.Path(__file__).with_name('record_live_greeks_web.cjs');record=subprocess.Popen(['node',str(recorder),url,str(out)],stdout=open(out/'recording.log','wb'),stderr=subprocess.STDOUT)
   for _ in range(200):
    if (out/'browser-ready.json').exists():break
    if record.poll() is not None:raise RuntimeError('recorder exited early')
    time.sleep(.05)
   else:raise RuntimeError('browser readiness deadline exceeded')
  with open(out/'input-frames.jsonl','wb') as inputs, open(out/'snapshots.jsonl','wb') as history:
   history_bytes=0
   for i in range(a.frames):
    f,truth=frame(i,start,a.interval);inputs.write(json.dumps(f,separators=(',',':')).encode()+b'\n')
    if i:
     v,h=http(url+'/api/quotes',f);receipts.append(dict(check(v,f,truth),snapshot_sha256=h))
    payload=json.dumps(v,separators=(',',':')).encode()+b'\n';history_bytes+=len(payload);assert history_bytes<=64*1024*1024;history.write(payload);history.flush()
    write(out/'status.json',dict(state='replaying',generation=i+1,url=url,server_pid=proc.pid));time.sleep(a.interval)
  # The same generation must be rejected and leave publication unchanged.
  try:http(url+'/api/quotes',f);raise AssertionError('duplicate generation accepted')
  except urllib.error.HTTPError as e:assert e.code==422
  final,_=http(url+'/api/snapshot');assert final['generation']==a.frames and final['input_sha256']==v['input_sha256'];write(out/'last-snapshot.json',final)
  time.sleep(.4);assert not stream_errors and events and events[-1]['generation']==a.frames
  assert all(x['generation']<y['generation'] for x,y in zip(events,events[1:]));assert all(r['source_not_after_calculation'] for r in receipts),'future source clock'
  bygen={r['generation']:r['input_sha256'] for r in receipts};assert all(bygen[e['generation']]==e['input_sha256'] for e in events)
  if record:assert record.wait(timeout=45)==0,'browser recording failed; inspect recording.log'
  write(out/'receipts.json',dict(status='pass',frames=receipts,sse_events=events,stale_generation_rejected=True,final_publication_preserved=True,max_price_error=max(r['max_price_error'] for r in receipts),max_delta_error=max(r['max_delta_error'] for r in receipts),actual_service_url=url))
  write(out/'status.json',dict(state='complete',generation=a.frames,url=url,server_pid=proc.pid,linger_seconds=a.linger_seconds));print(json.dumps(dict(status='pass',frames=a.frames,url=url,output=str(out))),flush=True)
  time.sleep(a.linger_seconds)
 finally:
  stop.set()
  if record and record.poll() is None:record.terminate();record.wait(timeout=5)
  if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=5)
  manifest={str(p.relative_to(out)):sha(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name!='artifact-manifest.json'};write(out/'artifact-manifest.json',manifest)
if __name__=='__main__':main()
