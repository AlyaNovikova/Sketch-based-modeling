import bpy, bpy_extras, sys, os

from mathutils import *
from math import *

scene = bpy.context.scene
scene.frame_start = 60
scene.frame_end = 62

bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 0.03 * pi])
bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 0.6 * pi])

# bpy.ops.object.camera_add(align='VIEW', rotation=[pi / 2, 0, 0.95 * pi])
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
        # ob.hide_render = True

    if ob.type == 'CAMERA':
        cameras.append(ob)

render_size = (
    int(scene.render.resolution_x),
    int(scene.render.resolution_y),
)
for frame in range(scene.frame_start, scene.frame_end):
    bpy.context.scene.frame_set(frame)
    for camera_num, camera in enumerate(cameras):
        scene.camera = camera
        bpy.ops.view3d.camera_to_view_selected()

        f = open(f'/home/alya/work/sketch/dataset/renders/skeleton/{camera_num}_{frame}', 'w')
        for arm in scene.objects:
            if arm.name == 'Armature.001':
                for i, b in enumerate(arm.pose.bones):
                    if is_part_of_skeleton(b.name, skeleton_bones, not_skeleton_bones):
                        global_location = arm.matrix_world @ b.head
                        coords_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, global_location)

                        coords_pixel = (
                            round(coords_2d[0] * render_size[0]),
                            round((1 - coords_2d[1]) * render_size[1]),
                        )
                        f.write(f'{b.name} {coords_pixel[0]} {coords_pixel[1]}\n')
                        # f.write(f'{b.name} {coords_2d[0] / coords_2d[2]} {coords_2d[1] / coords_2d[2]}\n')
        f.write('\n')
        f.close()

        file = os.path.join('/home/alya/work/sketch/dataset/renders/', f'{camera_num}_{frame}')
        bpy.context.scene.render.filepath = file
        bpy.ops.render.render(write_still=True)

for camera in cameras:
    bpy.data.objects.remove(camera)

print('Done!')