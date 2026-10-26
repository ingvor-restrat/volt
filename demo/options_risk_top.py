#!/usr/bin/env python3
"""Terminal view of exercise-risk snapshots from a worker URL or local JSONL file."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import os


def render(snapshot, max_rows=16):
    rows = snapshot['rows']
    totals = {k:0. for k in ['value','delta_dollars','gamma_1pct','vega_per_vol_point','theta_per_day']}
    for row in rows:
        units=row.get('quantity',0)*row.get('multiplier',100)
        if not units:continue
        risk=row.get('risk')
        if risk is None:
            totals={k:None for k in totals};break
        values={'value':risk['valuation']['price'],'delta_dollars':risk['delta']*row['spot'],'gamma_1pct':.5*risk['gamma']*(.01*row['spot'])**2,'vega_per_vol_point':risk['vega_per_vol_point'],'theta_per_day':risk.get('theta_per_day')}
        for key,value in values.items():
            if value is None:totals[key]=None
            elif totals[key] is not None:totals[key]+=units*value
    def num(v, width=13, digits=2):
        return ('—' if v is None else f'{v:,.{digits}f}').rjust(width)
    def exposure(items):
        return sum(.5 * r.get('quantity',0) * r.get('multiplier',100) * r['risk']['gamma'] * (.01*r['spot'])**2 for r in items if r.get('risk'))
    line='─'*126
    output=['VOLT  /  OPTIONS RISK                                                   '+snapshot['source'].upper(),
            f"Observation ns {snapshot.get('as_of_ns','—')}   ·   Generation {snapshot['generation']}   ·   {len(rows)} contracts",
            'Per-unit contract Greeks · portfolio quantities in contracts × multiplier',line,
            '          VALUE       DELTA DOLLARS       GAMMA / 1%       VEGA / POINT       THETA / DAY',
            ''.join(num(totals.get(k),18) for k in ['value','delta_dollars','gamma_1pct','vega_per_vol_point','theta_per_day']),line,
            'UNDERLYING EXPOSURES · gamma price contribution for a 1% stock move']
    symbols=sorted(set(r['underlying'] for r in rows));values={s:exposure([r for r in rows if r['underlying']==s]) for s in symbols}
    scale=max([abs(v) for v in values.values()]+[1e-9])
    for symbol,value in values.items():
        output.append(f'{symbol:<7} '+('█'*round(abs(value)/scale*36)).ljust(38)+num(value))
    output += [line,'SYMBOL  RIGHT STRIKE   EXPIRY       EXERCISE       STOCK       BID / ASK         IV     DELTA     GAMMA    VEGA/PT  THETA/DAY']
    for row in rows[-max_rows:]:
        risk=row.get('risk') or {};iv=(row.get('iv') or {}).get('volatility')
        from datetime import datetime, timezone
        expiry=datetime.fromtimestamp(int(row['expiry_ns'])/1e9,timezone.utc).strftime('%Y-%m-%d')
        output.append(f"{row['underlying']:<7} {row['side'][0].upper()} {row['strike']:>8.2f}  {expiry}  {row['exercise_style']:<10} {row['spot']:>9.2f}   {row['bid']:>6.2f}/{row['ask']:<6.2f} "+num(None if iv is None else iv*100,7)+num(risk.get('delta'),10,3)+num(risk.get('gamma'),10,4)+num(risk.get('vega_per_vol_point'),10,3)+num(risk.get('theta_per_day'),10,3))
    output += [line,f"Showing latest {min(max_rows,len(rows))} arrivals · calculation {snapshot.get('calculation_ms',0):,.1f} ms · {sum(bool(r.get('risk')) for r in rows)}/{len(rows)} calculated"]
    assumptions=snapshot.get('assumptions',{})
    if isinstance(assumptions.get('positions'),str):output.append(assumptions['positions'])
    return '\n'.join(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path);source.add_argument('--url')
    parser.add_argument('--interval',type=float,default=1.)
    parser.add_argument('--rows',type=int,default=16)
    parser.add_argument('--record',action='store_true',help='separate snapshots with form-feed for a renderer')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 1 <= args.rows <= 128 or not .02 <= args.interval <= 60:parser.error('rows 1..128 and interval .02..60 seconds')
    if args.input:
        if args.input.stat().st_size>25*1024*1024:parser.error('replay exceeds 25 MiB')
        frames=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        if len(frames)>2000:parser.error('replay exceeds 2000 snapshots')
    else:frames=None
    i=0
    while True:
        if frames is not None:
            if i>=len(frames):break
            snapshot=frames[i]
        else:
            headers={};token=os.environ.get('VOLT_RISK_TOKEN')
            if token:headers['Authorization']='Bearer '+token
            with urlopen(Request(args.url.rstrip('/')+'/api/options-risk/snapshot',headers=headers),timeout=10) as response:
                snapshot=json.loads(response.read(25*1024*1024+1))
        if snapshot.get('schema')!='volt.options_risk_snapshot.v1':raise ValueError('unsupported risk snapshot')
        print(('\f' if i else '') if args.record else '\033[2J\033[H',end='')
        print(render(snapshot,args.rows),flush=True);i+=1
        if args.once:break
        if not args.record:time.sleep(args.interval)

if __name__=='__main__':main()
