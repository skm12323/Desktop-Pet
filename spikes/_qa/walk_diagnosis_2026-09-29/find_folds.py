import json, sys
import numpy as np
sys.path.insert(0,'.')
from pet.rig.skinned_mesh_item import RigRuntime
from pet.rig.gait import GaitSolver
pkg='assets/rig_adult_walk_v1'
spec=json.load(open(f'{pkg}/spec.json',encoding='utf8'))
rt=RigRuntime.load(f'{pkg}/spec.json',f'{pkg}/mesh/mesh_data.json',f'{pkg}/layers')
ids=sys.argv[1].split(',') if len(sys.argv)>1 else ['leg_l','leg_r']
layers=[l for l in rt.layers if l.layer_id in ids]
def areas(p,t):
    a,b,c=(p[t[:,i],:2] for i in range(3)); return (b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
summary={}
for speed in (120,200):
    g=GaitSolver(spec,256/1696)
    for i in range(480):
        o=g.update(1/60,speed if i<300 else 0,(round(g.window_x_float),0))
        ang=np.array([np.degrees(o.bone_rotations.get(b.name,0)) for b in rt.bones],np.float32)
        tx,ty=np.zeros_like(ang),np.zeros_like(ang); tx[rt.bone_index['root_hip']],ty[rt.bone_index['root_hip']]=o.pelvis_offset
        rt.skinning_matrices(ang,tx,ty,0,0)
        for L in layers:
            tri=L.triangles.reshape(-1,3); d=rt.deform(L,L.rest); bad=np.nonzero(areas(L.rest,tri)*areas(d,tri)<0)[0]
            if len(bad):
                k=(speed,L.layer_id); summary.setdefault(k,[0,set()]); summary[k][0]+=1
                for c in np.round(L.rest[tri[bad]].mean(1)[:,:2]).tolist(): summary[k][1].add(tuple(c))
for k,(n,locs) in summary.items(): print(k,'frames with folds',n,'locations',sorted(locs)[:10])
if not summary: print('no folds in', ids)
