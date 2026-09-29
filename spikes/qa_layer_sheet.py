"""Contact sheet of rig layers on a checker background (G4 evidence)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

d, out = Path(sys.argv[1]), sys.argv[2]
ids = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(p.stem for p in d.glob("*.png"))
tw, th = 240, 424
chk = Image.new("RGBA", (960, 1696), (255, 255, 255, 255))
dr = ImageDraw.Draw(chk)
for y in range(0, 1696, 32):
    for x in range(0, 960, 32):
        if (x // 32 + y // 32) % 2:
            dr.rectangle([x, y, x + 31, y + 31], fill=(210, 214, 224, 255))
cols = 8
sheet = Image.new("RGB", (tw * cols, th * ((len(ids) + cols - 1) // cols)), "white")
for k, i in enumerate(ids):
    im = chk.copy()
    im.alpha_composite(Image.open(d / f"{i}.png").convert("RGBA"))
    t = im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
    ImageDraw.Draw(t).text((4, 4), i, fill=(200, 0, 0))
    sheet.paste(t, ((k % cols) * tw, (k // cols) * th))
sheet.save(out)
