"""Read-only visual audit; writes images/measurements only into this directory."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
from PIL import Image, ImageDraw
from render_rig_rest import RigRenderer
from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionEngine, MotionInputs
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

OUT = Path(__file__).parent

def bg(im, color=(244, 246, 250)):
    b = Image.new("RGBA", im.size, (*color, 255))
    b.alpha_composite(im)
    return b.convert("RGB")

def sheet(tiles, cols, name):
    w = max(im.width for _, im in tiles)
    h = max(im.height for _, im in tiles) + 22
    out = Image.new("RGB", (w*cols, h*((len(tiles)+cols-1)//cols)), "white")
    d = ImageDraw.Draw(out)
    for i, (label, im) in enumerate(tiles):
        x, y = (i%cols)*w, (i//cols)*h
        out.paste(im, (x,y+22))
        d.text((x+4,y+4), label, fill="black")
    out.save(OUT / name)

def grab(r):
    r._pump()
    q = r.win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
    arr = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine()//4,4)[:,:q.width()].copy()
    im = Image.fromarray(arr, "RGBA")
    return im.resize((r.win.width(), r.win.height()), Image.Resampling.LANCZOS)

def main():
    r = RigRenderer(stage="final")
    report = {}
    try:
        rest = r.render(r.rest_frame())
        rest.save(OUT / "rest_canvas.png")
        key = Image.open(ROOT / "assets/rig_final/references/front_key.png").convert("RGBA")
        a, k = np.array(rest,dtype=np.float32), np.array(key,dtype=np.float32)
        premul = np.abs(a[:,:,:3]*a[:,:,3:] - k[:,:,:3]*k[:,:,3:]).max(-1)/255
        op = k[:,:,3]>0
        report["rest_vs_key"] = {"mean_255": float(premul[op].mean()), "fraction_gt24": float((premul[op]>24).mean()), "full_rgba_bbox":rest.getbbox()}
        small = r.render(r.rest_frame(), size=(320,320), ground_shift=True)
        small.save(OUT / "rest_320.png")
        sheet([(label,bg(small,c)) for label,c in [("320 actual / light",(244,246,250)),("320 actual / dark",(32,36,43))]],2,"rest_backgrounds.png")
        crops = [("face",(415,180,620,350)),("left wrist",(225,775,350,960)),("right wrist",(680,775,810,960)),("feet",(425,1540,610,1780)),("tail",(630,1180,1015,1630))]
        sheet([(name,bg(rest.crop(box)).resize((int((box[2]-box[0])*1.5),int((box[3]-box[1])*1.5)),Image.Resampling.NEAREST)) for name,box in crops],3,"rest_details.png")
        eyes = (430,210,610,285)
        tiles=[]
        for b in (0,.5,1,0):
            f=r.rest_frame();f.blink_progress=b
            im=r.render(f)
            tiles.append((f"blink {b}",bg(im.crop(eyes)).resize((540,225),Image.Resampling.NEAREST)))
        sheet(tiles,2,"blink_review.png")
        tiles=[]
        for label,look in [("centre",(0,0)),("left",(-1,0)),("right",(1,0)),("up",(0,-1)),("down",(0,1))]:
            f=r.rest_frame();f.look_at=look
            im=r.render(f)
            tiles.append((label,bg(im.crop(eyes)).resize((540,225),Image.Resampling.NEAREST)))
        sheet(tiles,3,"gaze_review.png")

        tiles=[]; states=[]
        for branch in ("healthy","neglected"):
            for mood in ("neutral","happy","sad","hungry","sleepy"):
                path=ROOT / "assets/ai" / f"final_{branch}_{mood}.png"
                r.win.set_sprite(SpriteRef(str(path),320,320))
                im=r.render(r.rest_frame(),size=(320,320),ground_shift=True)
                im.save(OUT / f"state_{branch}_{mood}.png")
                tiles.append((f"{branch}/{mood}",bg(im)))
                states.append({"state":f"{branch}_{mood}","activeFigure":r.win._root.property("activeFigure"),"skinnedVisible":r.win._root.property("skinnedMeshVisible"),"bbox":im.getbbox()})
        sheet(tiles,5,"states_320.png")
        report["states"]=states

        r.win.set_sprite(SpriteRef(str(ROOT/"assets/ai/final_healthy_neutral.png"),320,320))
        r.win.resize(320,320);r._pump()
        engine=MotionEngine(r.spec)
        inp=MotionInputs(walking=False,walk_hz=0,facing=1,grounded=True,cursor_pos=(160,60),pet_rect=(0,0,320,320))
        frames=[]; measurements=[]; tiles=[]
        for i in range(150):
            f=engine.step(inp,66)
            r.win._root.setProperty("skinnedGroundYPx",r.ground_y)
            r.win._push_frame(f)
            QTest.qWait(66)
            im=grab(r)
            if i%2==0:frames.append(bg(im))
            # Shift the character up to retain the pixels normally below the viewport.
            r.win._root.setProperty("bodyY",f.body_y-20)
            up=grab(r)
            aa=np.asarray(up)[250:,:,3]
            ys,xs=np.where(aa>128)
            sole_bottom=int(ys.max()+250+20) if len(ys) else None
            measurements.append({"time_ms":(i+1)*66,"body_y":f.body_y,"body_angle":f.body_angle,"sole_bottom_unclipped":sole_bottom,"feet_alpha_below_viewport_px":int((np.asarray(up)[300:,:,3]>128).sum())})
            if i in (11,35,60,84,108,132):
                im.save(OUT / f"idle_{i:03}.png")
                tiles.append((f"t={(i+1)*.066:.2f}s / y={f.body_y:.2f}",bg(im.crop((130,265,195,320))).resize((260,220),Image.Resampling.NEAREST)))
        frames[0].save(OUT/"idle_actual_320.gif",save_all=True,append_images=frames[1:],duration=132,loop=0)
        sheet(tiles,3,"feet_idle_review.png")
        bottoms=[m["sole_bottom_unclipped"] for m in measurements]
        report["idle_ground"]={"window_size":[320,320],"min_sole_bottom":min(bottoms),"max_sole_bottom":max(bottoms),"body_y_range":[min(m["body_y"] for m in measurements),max(m["body_y"] for m in measurements)],"clipped_frames":sum(m["feet_alpha_below_viewport_px"]>0 for m in measurements),"max_clipped_opaque_px":max(m["feet_alpha_below_viewport_px"] for m in measurements),"frames":measurements}
    finally:
        r.close()
    (OUT/"measurements.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!="frames"} if isinstance(v,dict) else v) for k,v in report.items()},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
