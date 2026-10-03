from pathlib import Path
import sys,json,math
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'spikes')]
from render_rig_rest import RigRenderer
from pet.rig.motion import MotionFrame
from pet.rig.gait import GaitSolver
from qa_final_side_rig import POSES
PKG=ROOT/'assets/rig_final_walk_v1'

def light(im):
    out=Image.new('RGBA',im.size,(245,246,250,255));out.alpha_composite(im);return out.convert('RGB')
def sheet(tiles,cols,name):
    w=max(im.width for _,im in tiles);h=max(im.height for _,im in tiles)+24
    out=Image.new('RGB',(w*cols,h*math.ceil(len(tiles)/cols)),(245,246,250));d=ImageDraw.Draw(out)
    for i,(name0,im) in enumerate(tiles):
        x=i%cols*w;y=i//cols*h;out.paste(light(im),(x,y+24));d.text((x+4,y+4),name0,fill='black')
    out.save(OUT/name)

r=RigRenderer('final',str(PKG/'final'))
tiles=[]
try:
    rest=r.render(r.rest_frame());rest.save(OUT/'side_rest_after.png')
    old=Image.open(PKG/'references/side_rest.png').convert('RGBA')
    a,b=np.asarray(rest,dtype=float),np.asarray(old,dtype=float)
    diff=np.abs(a[:,:,:3]*a[:,:,3:]-b[:,:,:3]*b[:,:,3:])/255
    fg=(a[:,:,3]>8)|(b[:,:,3]>8)
    metrics={'rest_vs_previous_mean_255':float(diff.mean(-1)[fg].mean()),'max_255':float(diff.max())}
    for name,angles in [('rest',{}),('tail_sway',POSES['dbg_tail']),('idle_sway',POSES['idle_sway'])]:
        f=r.rest_frame();f.bone_angles.update(angles);im=r.render(f)
        im.save(OUT/f'{name}_after.png')
        for part,box in [('fluke',(210,860,420,1120)),('rear_skirt',(240,930,430,1130))]:
            tiles.append((name+' '+part,im.crop(box).resize((420,520))))
    sheet(tiles,3,'poses_after.png')
    spec=json.loads((PKG/'spec.json').read_text());g=GaitSolver(spec,320/1824)
    frames=[];phases=[]
    for i in range(90+round(2/spec['gait']['frequency_hz']*60)):
        o=g.update(1/60,80,(round(g.window_x_float),0))
        if i<90 or i%2:continue
        ang={b:math.degrees(v) for b,v in o.bone_rotations.items()}
        tx={'root_hip':o.pelvis_offset[0]};ty={'root_hip':o.pelvis_offset[1]}
        for bone,(ox,oy) in o.bone_offsets.items():tx[bone]=tx.get(bone,0)+ox;ty[bone]=ty.get(bone,0)+oy
        f=r.rest_frame();f.bone_angles.update(ang);f.bone_tx=tx;f.bone_ty=ty
        im=r.render(f)
        frames.append(light(r.render(f,size=(320,320),ground_shift=True)))
        if (i-90)%8==0:phases.append((str(i),im.crop((150,830,500,1320)).resize((350,490))))
    sheet(phases,4,'walk_detail_after.png')
    # GIF delays are centiseconds. Distribute 30/40 ms delays so sampling
    # every second 60 Hz tick stays at 30 fps instead of being sped up.
    delays=[10*(round((i+1)*100/30)-round(i*100/30)) for i in range(len(frames))]
    frames[0].save(OUT/'walk_after_320.gif',save_all=True,append_images=frames[1:],duration=delays,loop=0)
    frames[0].save(OUT/'walk_after_320_slow.gif',save_all=True,append_images=frames[1:],duration=delays,loop=0)
finally:r.close()
sheet([('before',Image.open(OUT/'tail_before.png').convert('RGBA')),
       ('after',Image.open(OUT/'idle_sway_after.png').crop((0,820,440,1380)).resize((880,1120)))],2,'tail_before_after.png')
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
