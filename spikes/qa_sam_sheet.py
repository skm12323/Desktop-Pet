"""Overlay sheet for a folder of SAM masks on the side key art (G4 evidence)."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

key = Image.open('assets/rig_adult_walk_v1/references/side_key.png').convert('RGBA')
bg = Image.new('RGBA', key.size, (255, 255, 255, 255))
bg.alpha_composite(key)
base = np.asarray(bg.convert('RGB'), np.float32)
d, out = Path(sys.argv[1]), sys.argv[2]
files = sorted(p for p in d.glob('*.png'))
tiles = []
for p in files:
    m = np.asarray(Image.open(p).convert('L')) > 127
    v = base.copy()
    v[m] = v[m] * 0.45 + np.array([255, 0, 0]) * 0.55
    im = Image.fromarray(v.astype(np.uint8)).resize((240, 424))
    ImageDraw.Draw(im).text((4, 4), p.stem, fill=(0, 0, 0))
    tiles.append(im)
cols = 8
sheet = Image.new('RGB', (240 * cols, 424 * ((len(tiles) + cols - 1) // cols)), 'white')
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols) * 240, (k // cols) * 424))
sheet.save(out)
