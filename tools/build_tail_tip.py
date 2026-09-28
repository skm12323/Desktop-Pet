import cv2
from PIL import Image
import numpy as np

ref = Image.open('assets/reference/adult_ref.png')
ref_rgb = np.array(ref)

# Full canvas 960x1696
h, w, _ = ref_rgb.shape

# 1. Background mask
is_bg = (ref_rgb[:, :, 0] > 242) & (ref_rgb[:, :, 1] > 242) & (ref_rgb[:, :, 2] > 242)

# 2. Hand exclusion (hand is at X in [680, 770], Y in [850, 930])
# We can define a polygon or curve for the top edge of the fluke:
# Top tip of right lobe is around (804, 875)
# Notch is at (782, 985)
# Top of left lobe is at (725, 935)
# Let's inspect the exact line of the left lobe's top edge
# Any pixel with Y < 930 and X < 765 is hand / cuff
is_hand = (np.arange(w)[None, :] < 765) & (np.arange(h)[:, None] < 935)

# Also exclude skirt in adult_ref:
skirt = Image.open('assets/rig_adult/layers/skirt.png')
skirt_a = np.array(skirt)[:, :, 3]
is_skirt = skirt_a > 50

# Fluke region: X in [680, 920], Y in [870, 1100]
in_tail_roi = (np.arange(w)[None, :] >= 680) & (np.arange(w)[None, :] <= 920) & \
              (np.arange(h)[:, None] >= 870) & (np.arange(h)[:, None] <= 1085)

tail_vis = in_tail_roi & (~is_bg) & (~is_hand) & (~is_skirt)

# Left lobe extension under skirt:
# The left lobe in adult_ref has an upper curve extending towards X=690, Y=970
# and a lower curve extending towards X=740, Y=1030
# Let's create a rounded left lobe polygon or ellipse under the skirt:
# Center around (730, 980), radius_x = 40, radius_y = 35
yy, xx = np.ogrid[:h, :w]
left_lobe_ellipse = ((xx - 720)**2 / (45**2) + (yy - 980)**2 / (38**2)) <= 1
# Clip to under the skirt and not background
left_lobe_ext = left_lobe_ellipse & is_skirt & in_tail_roi

# Combine tail mask
full_tail_mask = tail_vis | left_lobe_ext

# Create RGBA for tail_tip
tail_tip = np.zeros((h, w, 4), dtype=np.uint8)
tail_tip[tail_vis, :3] = ref_rgb[tail_vis]
tail_tip[full_tail_mask, 3] = 255

# For left_lobe_ext (under the skirt), inpaint from the visible fluke
inpaint_mask = (left_lobe_ext & (tail_tip[:, :, 3] > 0)).astype(np.uint8) * 255
# Pre-fill with fluke navy blue
fluke_navy = np.array([55, 75, 125], dtype=np.uint8)
tail_tip_rgb = tail_tip[:, :, :3].copy()
tail_tip_rgb[left_lobe_ext] = fluke_navy

bgr = cv2.cvtColor(tail_tip_rgb, cv2.COLOR_RGB2BGR)
inpainted_bgr = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
tail_tip[:, :, :3] = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

# Anti-alias the outer boundary of tail_tip against background
# Find outer boundary (edge with alpha=0)
alpha = tail_tip[:, :, 3].copy()
alpha_blurred = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0.5)
tail_tip[:, :, 3] = np.clip(alpha_blurred, 0, 255).astype(np.uint8)

# Save preview of tail_tip alone
Image.fromarray(tail_tip).crop((680, 860, 920, 1100)).save('assets/rig_adult/debug_new_tail_tip.png')

# Now let's test composite with tail_seg2 and skirt!
s2 = Image.open('assets/rig_adult/layers/tail_seg2.png')

comp = Image.new('RGBA', (w, h), (0, 0, 0, 0))
comp.alpha_composite(s2)
comp.alpha_composite(Image.fromarray(tail_tip))
comp.alpha_composite(skirt)

comp.crop((680, 860, 920, 1150)).save('assets/rig_adult/debug_tail_comp_test.png')
print('Saved debug_new_tail_tip.png and debug_tail_comp_test.png')
