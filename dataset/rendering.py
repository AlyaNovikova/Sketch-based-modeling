import bpy, bpy_extras, sys, os

from math import *

import sys

bpy.data.scenes[0].render.engine = "CYCLES"

# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU
    print(d["name"], d["use"])

scene = bpy.context.scene
for ob in scene.objects:
    bpy.data.objects.remove(ob)

model_root = '/home/sketch/animations/models/'
root = '/home/sketch/animations/dataset/'
# characters = ['abe', 'eve', 'jennifer', 'lola', 'mannequin', 'xbot']
characters = [sys.argv[-1]]
global_cnt = 0
for character_name in characters:
    animation_dir = model_root + character_name + '/regular/'
    for animation_i, animation_file in enumerate(os.listdir(animation_dir)):
        bpy.ops.import_scene.fbx(filepath = animation_dir + animation_file, use_anim=True)
        animation = bpy.context.object
        bpy.ops.import_scene.fbx(filepath = model_root + character_name + '/' + character_name + '.fbx')
        model = bpy.context.object

        for ob in scene.objects:
            if ob.type == 'ARMATURE' or ob.type == 'MESH':
                ob.select_set(True)

        bpy.context.view_layer.objects.active = animation

        bpy.ops.object.make_links_data(type='ANIMATION')

        for i in range(8):
            bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 2 * pi * i / 8])
            # bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 5, 0, 2 * pi * i / 8])

        # bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 0.03 * pi])
        # bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2 * 0.9, 0.05, 0.75 * pi])
        # bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2 * 0.6, -0.1, 0.95 * pi])
        # bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 1.36 * pi])

        def good_bone_name(x):
            x = x[10:]
            if 'Left' in x:
                x = x[4:] + ".L"
            if 'Right' in x:
                x = x[5:] + ".R"
            return x


        def is_part_of_skeleton(name, skeleton_bones, not_skeleton_bones):
            for b in skeleton_bones:
                if b in name:
                    flag = True
                    for nb in not_skeleton_bones:
                        if nb in name:
                            flag = False
                            break
                    if flag:
                        return True
            return False


        skeleton_bones = ['Hips',
                          'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
                          'Shoulder', 'Arm', 'ForeArm', 'Hand',
                          'UpLeg', 'Leg', 'Foot']

        not_skeleton_bones = ['HandThumb', 'HandIndex', 'HandMiddle', 'HandRing', 'HandPinky', ]

        cameras = []
        for ob in scene.objects:
            if ob.type == 'ARMATURE' or ob.type == 'MESH':
                ob.select_set(True)
                if ob.type == 'MESH':
                    mat = bpy.data.materials.new(name="Material")

                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    for node in nodes:
                        nodes.remove(node)
                    links = mat.node_tree.links

                    node_output = nodes.new(type='ShaderNodeOutputMaterial')
                    node_output.location = 0, 0
                    node_output.target = 'CYCLES'

                    # node_pbsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                    # node_pbsdf.location = 0, 0

                    node_pbsdf = nodes.new(type='ShaderNodeHoldout')
                    node_pbsdf.location = 0, 0

                    # node_pbsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 0.0)
                    # node_pbsdf.inputs['Alpha'].default_value = 0
                    # node_pbsdf.inputs['Roughness'].default_value = 1
                    # node_pbsdf.inputs['Transmission'].default_value = 1
                    # node_pbsdf.inputs['Transmission Roughness'].default_value = 0
                    # node_pbsdf.inputs['Transmission Roughness'].default_value = 0

                    link = links.new(node_pbsdf.outputs['Holdout'], node_output.inputs['Surface'])

                    mat.blend_method = 'CLIP'
                    mat.shadow_method = 'CLIP'
                    mat.use_screen_refraction = True

                    if ob.data.materials:
                        ob.data.materials[0] = mat
                        ob.active_material = mat
                    else:
                        ob.data.materials.append(mat)

                    ob.active_material.alpha_threshold = 1000
                    ob.active_material.diffuse_color = (0, 0, 0, 0)
                    ob.active_material.blend_method = 'CLIP'
                    ob.active_material.shadow_method = 'CLIP'

                    ob.active_material.roughness = 0

                    ob.active_material.use_backface_culling = True
                    ob.active_material.use_sss_translucency = True
                    ob.active_material.show_transparent_back = True

                else:
                    ob.hide_render = True

            if ob.type == 'CAMERA':
                cameras.append(ob)

        freestyle = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
        scene.render.use_freestyle = True
        scene.render.line_thickness = 1

        linesets = freestyle.linesets.new('VisibleLineset')

        linesets.select_by_visibility = True
        linesets.select_by_edge_types = True
        linesets.select_by_image_border = True

        linesets.select_silhouette = True
        linesets.select_border = True
        linesets.select_crease = True

        def get_keyframes(obj_list):
            keyframes = 5
            for obj in obj_list:
                anim = obj.animation_data
                if anim is not None and anim.action is not None:
                    for fcu in anim.action.fcurves:
                        for keyframe in fcu.keyframe_points:
                            x, y = keyframe.co
                            keyframes = ceil(x)
            return keyframes


        selection = bpy.context.selected_objects
        key = get_keyframes(selection)
        scene.frame_start = 1
        scene.frame_end = key

        scene.frame_start = 1
        # if scene.frame_end > 5:
        #     scene.frame_end = 2

        save_res_x = scene.render.resolution_x
        save_res_y = scene.render.resolution_y

        scale = scene.render.resolution_y / 512
        scene.render.resolution_x /= scale
        scene.render.resolution_x *= 0.6
        scene.render.resolution_y /= scale
        render_size = (
            int(scene.render.resolution_x),
            int(scene.render.resolution_y),
        )

        for frame in range(scene.frame_start, scene.frame_end, 10):
            bpy.context.scene.frame_set(frame)
            for camera_num, camera in enumerate(cameras):
                scene.camera = camera
                bpy.ops.view3d.camera_to_view_selected()
                bpy.context.scene.camera.data.sensor_width = 39

                f = open(root + character_name + f'/skeleton/{animation_i}_{camera_num}_{frame}_{global_cnt}', 'w')
                for arm in scene.objects:
                    if arm.name == model.name:
                        for i, b in enumerate(arm.pose.bones):
                            if is_part_of_skeleton(b.name, skeleton_bones, not_skeleton_bones):
                                global_location = arm.matrix_world @ b.head
                                coords_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, global_location)

                                coords_pixel = (
                                    round(coords_2d[0] * render_size[0]),
                                    round((1 - coords_2d[1]) * render_size[1]),
                                )
                                f.write(f'{b.name} {coords_pixel[0]} {coords_pixel[1]}\n')
                f.write('\n')
                f.close()

                file = os.path.join(root + character_name + '/', f'{animation_i}_{camera_num}_{frame}_{global_cnt}')
                global_cnt += 1

                # img = bpy.data.images.load(root + 'paper.jpg')
                # bpy.context.scene.camera.data.show_background_images = True
                # bg = bpy.context.scene.camera.data.background_images.new()
                # bg.image = img
                #
                # for area in bpy.context.screen.areas:
                #     if area.type == 'VIEW_3D':
                #         override = bpy.context.copy()
                #         override['area'] = area
                #         bpy.ops.view3d.background_image_add(override, name="BG", filepath=root + 'paper.jpg')
                #         break

                scene.render.film_transparent = True

                scene.render.filepath = file
                bpy.ops.render.render(write_still=True)
                bpy.context.scene.camera.data.sensor_width = 36

        scene.render.resolution_x = save_res_x
        scene.render.resolution_y = save_res_y

        for ob in scene.objects:
           bpy.data.objects.remove(ob)

print('Done!')
print('Global cnt =', global_cnt)