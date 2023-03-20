"""Smooth all the meshes in the scene."""

import bpy

# Loop through all the objects in the scene
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            # Set shading mode to smooth
            obj.data.use_auto_smooth = True
            for poly in obj.data.polygons:
                poly.use_smooth = True
            # Apply "Shade Smooth" operator to the object
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
