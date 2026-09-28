import cv2
from PIL import Image
import numpy as np

ref = Image.open('assets/reference/adult_ref.png')
ref_rgb = np.array(ref)

# Region of interest for tail_tip:
# Y in [870, 1120], X in [680, 920]
roi_y0, roi_y1 = 870, 1120
roi_x0, roi_x1 = 680, 920

sub_ref = ref_rgb[roi_y0:roi_y1, roi_x0:roi_x1]

# Non-white mask:
# Background is white: R>242, G>242, B>242
is_bg = (sub_ref[:, :, 0] > 242) & (sub_ref[:, :, 1] > 242) & (sub_ref[:, :, 2] > 242)

# Exclude hand: Hand is in upper left: y_sub < 60, x_sub < 70
# Hand has skin tones: R>235, G in [180, 220], B in [180, 220]
is_skin = (sub_ref[:, :, 0] > 230) & (sub_ref[:, :, 1] > 170) & (sub_ref[:, :, 2] > 170) & (sub_ref[:, :, 0] > sub_ref[:, :, 2] + 20)

# Exclude skirt:
skirt = Image.open('assets/rig_adult/layers/skirt.png')
sub_skirt = np.array(skirt)[roi_y0:roi_y1, roi_x0:roi_x1]
is_skirt = sub_skirt[:, :, 3] > 50

# Tail visible mask:
tail_vis = (~is_bg) & (~is_skin) & (~is_skirt)

# Save mask visualization
vis = np.zeros((roi_y1 - roi_y0, roi_x1 - roi_x0, 4), dtype=np.uint8)
vis[tail_vis, :3] = sub_ref[tail_vis]
vis[tail_vis, 3] = 255
Image.fromarray(vis).save('assets/rig_adult/debug_tail_vis_mask.png')
print('Saved debug_tail_vis_mask.png')
