"""Create an animation of a rotating mesh sequence from the current blender file.

Run in a terminal with:
rm -f /Users/adelemyers/code/my28brains/my28brains/results/tmp/* \
&& \
/Applications/Blender.app/Contents/MacOS/Blender \
-b /Users/adelemyers/code/my28brains/my28brains/blender/blends/0319_left_white_rotate.blend \
--background \
--python "/Users/adelemyers/code/my28brains/my28brains/blender/create_animation.py" \
&& \
ffmpeg -y \
-framerate 2 \
-i /Users/adelemyers/code/my28brains/my28brains/results/tmp/0319_substructure_-1_framerate_2_%04d.jpg \
-c:v libx264 \
-pix_fmt yuv420p \
/Users/adelemyers/code/my28brains/my28brains/results/anims/0319_substructure_-1_framerate_2.mp4 \
&& \
open /Users/adelemyers/code/my28brains/my28brains/results/anims/0319_substructure_-1_framerate_2.mp4

"""

import math
import os
from datetime import date

import bpy
from mathutils import Euler

today = date.today()

RESULTS_DIR = "/Users/adelemyers/code/my28brains/results"
TMP = os.path.join(RESULTS_DIR, "tmp")

scene = bpy.context.scene

framerate = 2
scene.render.fps = framerate
scene.frame_end = 30
work_day = "0319"
rotation_fact = 0  # angular velocity "factor".
rotation_cst = 90  # TRY CREATING NIMATION ON THE SIDE TO CONFIRM IF IT'S HIIPOCMAPYUS
# ^change this to see from different angle

# Set render settings
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 100
scene.render.ffmpeg.codec = "HUFFYUV"
scene.render.ffmpeg.audio_codec = "NONE"


for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            x_rotation = math.radians(rotation_fact * frame + rotation_cst)
            y_rotation = 0
            z_rotation = 0

            obj.rotation_euler = Euler((x_rotation, y_rotation, z_rotation), "XYZ")

    filename = f"{TMP}/{work_day}_substructure_-1_framerate_{framerate}_{frame:04d}.jpg"
    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)
