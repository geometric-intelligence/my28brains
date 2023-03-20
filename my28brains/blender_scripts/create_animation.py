"""Create an animation of a rotating mesh sequence from the current blender file.

Run in a terminal with:
rm /Users/ninamiolane/code/my28brains/results/tmp/* \
&& \
/Applications/Blender.app/Contents/MacOS/Blender \
-b /Users/ninamiolane/code/my28brains/results/blender/0319_left_white_rotate.blend \
--background \
--python "/Users/ninamiolane/code/my28brains/my28brains/blender_scripts/create_animation.py" \ 
&& \
ffmpeg -y \
-framerate 2 \
-i /Users/ninamiolane/code/my28brains/results/tmp/0319_left_white_rotate_%04d.jpg \
-c:v libx264 \
-pix_fmt yuv420p \
/Users/ninamiolane/code/my28brains/results/blender/0319_substructure_1_white_rotate.mp4 \
&& \
open /Users/ninamiolane/code/my28brains/results/blender/0319_substructure_1_white_rotate.mp4

"""

import math
import os
from datetime import date

import bpy
from mathutils import Euler

today = date.today()

RESULTS_DIR = "/Users/ninamiolane/code/my28brains/results"
TMP = os.path.join(RESULTS_DIR, "tmp")

scene = bpy.context.scene

# Set render settings
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 2
scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.quality = 100
scene.render.ffmpeg.codec = "HUFFYUV"
scene.render.ffmpeg.audio_codec = "NONE"
scene.frame_end = 30  # or the number of frames you want

# Loop through all the objects in the scene
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.animation_data:
            x_rotation = 0
            y_rotation = math.radians(10 * frame + 60)
            z_rotation = 0

            obj.rotation_euler = Euler((x_rotation, y_rotation, z_rotation), "XYZ")

    filename = f"{TMP}/0319_left_white_rotate_{frame:04d}.jpg"
    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)
