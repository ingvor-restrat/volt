#!/usr/bin/env python3
"""Render actual native snapshot replay using the repository's terminal demo renderer."""
import argparse,pathlib,subprocess,importlib.util,hashlib,json
from PIL import ImageFont
P=argparse.ArgumentParser();P.add_argument('--snapshots',required=True,type=pathlib.Path);P.add_argument('--duration-ms',type=int,default=250);P.add_argument('--public',action='store_true');P.add_argument('--output',type=pathlib.Path,required=True);P.add_argument('--font',type=pathlib.Path);P.add_argument('--bold-font',type=pathlib.Path);a=P.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[2];O=a.output;O.mkdir(parents=True,exist_ok=True);exe=ROOT/'target/debug/examples/live_risk_top'
source=a.snapshots.read_bytes();snapshots=[json.loads(l) for l in source.splitlines() if l.strip()];n=len(snapshots);assert 1<=n<=1024
cmd=[str(exe),'--input',str(a.snapshots),'--record','--ticks',str(n),'--interval-ms','0']
r=subprocess.run(cmd,capture_output=True,check=True,timeout=30);assert subprocess.run(cmd,capture_output=True,check=True,timeout=30).stdout==r.stdout
(O/'frames.ansi.txt').write_bytes(r.stdout)
plain=subprocess.run([str(exe),'--input',str(a.snapshots),'--ticks','1','--interval-ms','0','--no-color'],capture_output=True,check=True,timeout=10).stdout;assert b'\x1b' not in plain;(O/'sample.txt').write_bytes(plain)
spec=importlib.util.spec_from_file_location('existing_terminal_renderer',ROOT/'docs/markup/record.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_spans=m.spans
def compatible_spans(line):
 for text,color,bold,dim in original_spans(line):
  yield text,color,(bold and '─' not in text),dim
m.spans=compatible_spans
if a.font:
 font=ImageFont.truetype(str(a.font),15);bold=ImageFont.truetype(str(a.bold_font or a.font),15)
elif pathlib.Path('/System/Library/Fonts/Menlo.ttc').exists():
 font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15,index=0);bold=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15,index=1)
else:
 font=ImageFont.truetype('DejaVuSansMono.ttf',15);bold=ImageFont.truetype('DejaVuSansMono-Bold.ttf',15)
frames=[m.render(f.strip('\n'),font,bold) for f in r.stdout.decode().split('\x0c') if f.strip()];assert len(frames)==n
for i,img in enumerate(frames):img.save(O/f'frame-{i:03}.png')
frames[0].save(O/'opening.png');frames[len(frames)//2].save(O/'middle.png');frames[-1].save(O/'closing.png')
gif=ROOT/'docs/markup/live-greeks-top.gif' if a.public else O/'live-greeks-top.gif'
frames[0].save(gif,save_all=True,append_images=frames[1:],duration=a.duration_ms,loop=0,disposal=2,optimize=True)
subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(1000/a.duration_ms),'-i',str(O/'frame-%03d.png'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',str(O/'live-greeks-top.mp4')],check=True,timeout=60)
manifest={'schema':'volt.live_greeks_top.recording.v1','source_snapshots':str(a.snapshots),'source_sha256':hashlib.sha256(source).hexdigest(),'source_frames':n,'pillow_version':__import__('PIL').__version__,'generation_range':[snapshots[0]['generation'],snapshots[-1]['generation']],'native_inputs_hashes':[s['input_sha256'] for s in snapshots],'snapshot_calculations':'Actual native service CPU IV/surface/Greeks/scenarios/Hypercube, replayed without recalculation in terminal','playback_frame_ms':a.duration_ms,'timing_scope':'Presentation pacing; saved service snapshots retain source cutoff/calculation timestamps. No latency measurement claimed.','identical_replay_stdout_verified':True,'command':cmd,'public_gif':str(gif),'artifacts_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [exe,ROOT/'examples/live_risk_top.rs',gif,O/'live-greeks-top.mp4',O/'sample.txt',O/'frames.ansi.txt',O/'opening.png',O/'middle.png',O/'closing.png']}}
(O/'recording-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(gif)
