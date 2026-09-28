import os
import shutil
import cv2
from PIL import Image
import numpy as np

# 0. Backup original tail_tip.png if backup doesn't exist
backup_dir = 'assets/rig_adult/backup_tail'
os.makedirs(backup_dir, exist_ok=True)
backup_path = os.path.join(backup_dir, 'tail_tip_orig.png')
if not os.path.exists(backup_path):
    shutil.copyfile('assets/rig_adult/layers/tail_tip.png', backup_path)
    print(f'Backed up original tail_tip.png to {backup_path}')

ref = Image.open('assets/reference/adult_ref.png')
ref_rgb = np.array(ref)
h, w, _ = ref_rgb.shape

skirt = Image.open('assets/rig_adult/layers/skirt.png')
skirt_a = np.array(skirt)[:, :, 3]

# 1. Background mask
is_bg = (ref_rgb[:, :, 0] > 240) & (ref_rgb[:, :, 1] > 240) & (ref_rgb[:, :, 2] > 240)

# 2. Skirt mask
is_skirt = (skirt_a > 10)

# 3. Hand exclusion
is_hand = (np.arange(w)[None, :] < 770) & (np.arange(h)[:, None] < 938)

# 4. Fluke ROI
fluke_roi = (np.arange(w)[None, :] >= 710) & (np.arange(w)[None, :] <= 920) & \
            (np.arange(h)[:, None] >= 840) & (np.arange(h)[:, None] <= 1080)

# Authentic visible fluke
fluke_vis = fluke_roi & (~is_bg) & (~is_skirt) & (~is_hand)

# 5. Extension under skirt for left lobe
left_lobe_pts = np.array([
    [726, 940],
    [695, 948],
    [670, 962],
    [655, 978],
    [660, 992],
    [685, 1002],
    [725, 1008],
    [766, 1005],
    [760, 980]
], dtype=np.int32)

ext_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(ext_mask, [left_lobe_pts], 255)
# Only where under skirt and not background or hand
ext_mask = (ext_mask > 0) & is_skirt & (~is_bg) & (~is_hand)

full_tail = fluke_vis | ext_mask

# Keep strictly the largest connected component to eliminate any fringe specks
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(full_tail.astype(np.uint8))
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
full_tail = (labels == largest_label)
fluke_vis = fluke_vis & full_tail

tail_tip = np.zeros((h, w, 4), dtype=np.uint8)
# Base navy color
tail_tip[full_tail, :3] = [54, 74, 122]
# Authentic reference pixels
tail_tip[fluke_vis, :3] = ref_rgb[fluke_vis]

# Add dark outline to extended lobe outer edge
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ext_eroded = cv2.erode(ext_mask.astype(np.uint8), kernel)
ext_edge = (ext_mask.astype(np.uint8) - ext_eroded).astype(bool)
tail_tip[ext_edge & (~fluke_vis) & full_tail, :3] = [28, 35, 65]

tail_tip[full_tail, 3] = 255

# Bottom blend into tail_seg2
for y in range(1060, 1080):
    t = (1080 - y) / 20.0
    tail_tip[y, :, 3] = (tail_tip[y, :, 3].astype(float) * t).astype(np.uint8)
tail_tip[1080:, :, 3] = 0

# Anti-alias alpha
alpha_f = tail_tip[:, :, 3].astype(np.float32)
alpha_blurred = cv2.GaussianBlur(alpha_f, (3, 3), 0.5)
tail_tip[:, :, 3] = np.clip(alpha_blurred, 0, 255).astype(np.uint8)

# Overwrite target layer
target_layer = 'assets/rig_adult/layers/tail_tip.png'
Image.fromarray(tail_tip).save(target_layer)
print(f'Successfully updated {target_layer}')

# Save verification crop
Image.fromarray(tail_tip).crop((640, 840, 920, 1100)).save('assets/rig_adult/debug_clean_tail_tip_final.png')

# Save full composite with tail_seg2 and skirt
s2 = Image.open('assets/rig_adult/layers/tail_seg2.png')
comp = Image.new('RGBA', (w, h), (0, 0, 0, 0))
comp.alpha_composite(s2)
comp.alpha_composite(Image.fromarray(tail_tip))
comp.alpha_composite(skirt)
comp.crop((640, 840, 920, 1150)).save('assets/rig_adult/debug_tail_comp_final.png')
print('Debug final generated successfully.')
