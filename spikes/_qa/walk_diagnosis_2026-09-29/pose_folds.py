import json, sys
import numpy as np
sys.path.insert(0,'.')
from pet.rig.skinned_mesh_item import RigRuntime
pkg='assets/rig_adult_walk_v1'
rt=RigRuntime.load(f'{pkg}/spec.json',f'{pkg}/mesh/mesh_data.json',f'{pkg}/layers')
def areas(p,t):
    a,b,c=(p[t[:,i],:2] for i in range(3)); return (b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
POSES={"near_leg_fwd": {"upper_leg_l": -25, "lower_leg_l": 30, "foot_l": -5, "upper_leg_r": 20,"lower_leg_r": 35, "foot_r": -55, "upper_arm_l": 20, "upper_arm_r": -20},
 "near_leg_back": {"upper_leg_l": 20, "lower_leg_l": 35, "foot_l": -55, "upper_leg_r": -25,"lower_leg_r": 30, "foot_r": -5, "upper_arm_l": -20, "upper_arm_r": 20},
 "passing": {"upper_leg_l": -25, "lower_leg_l": 60, "foot_l": -35, "upper_leg_r": 5,"lower_leg_r": 10, "foot_r": -15, "upper_arm_l": 5, "upper_arm_r": -5}}
extra=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
POSES.update(extra)
for name,pose in POSES.items():
    a=np.array([pose.get(b.name,0) for b in rt.bones],np.float32); z=np.zeros_like(a)
    rt.skinning_matrices(a,z,z,0,0)
    for L in rt.layers:
        t=L.triangles.reshape(-1,3); d=rt.deform(L,L.rest); bad=np.nonzero(areas(L.rest,t)*areas(d,t)<0)[0]
        if len(bad): print(name,L.layer_id,len(bad),np.round(L.rest[t[bad]].mean(1)[:,:2]).astype(int).tolist()[:8])
print('done')
