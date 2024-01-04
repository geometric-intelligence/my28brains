"""Contains a function to add a Subdivision Surface modifier to all meshes in the scene.

The function add_subdivision_modifier() loops through all the objects in the scene,
and adds a Subdivision Surface modifier to each one.

This module is part of the my28brains project.
"""

import bpy


def add_subdivision_modifier():
    """Add a Subdivision Surface modifier to all meshes in the scene.

    Loop through all the objects in the scene,
    and add a Subdivision Surface modifier to each one.

    Parameters
    ----------
    None

    Returns
    -------
    None

    TODO: merge this into create_animation.py by making it interact
    with the render.
    """
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            # Add Subdivision Surface modifier to the object
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_add(type="SUBSURF")
            bpy.context.object.modifiers["Subdivision"].levels = 2


add_subdivision_modifier()
