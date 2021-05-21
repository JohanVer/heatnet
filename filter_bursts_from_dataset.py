from glob import glob
import os


dataroot = '/media/zuern/Storage1/thermal/pub/train/'
fl_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fl_rgb/*.png')))
fr_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fr_rgb/*.png')))
fl_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fl_ir/*.png')))
fr_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fr_ir/*.png')))

# dataroot = '/media/zuern/Storage1/thermal/pub/train-dummy/'
# fl_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fl_rgb/*.png')))
# fr_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fr_rgb/*.png')))
# fl_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fl_ir/*.png')))
# fr_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fr_ir/*.png')))

print(len(fl_rgb_files))
print(len(fr_rgb_files))
print(len(fl_ir_files))
print(len(fr_ir_files))

assert len(fl_rgb_files) == len(fr_rgb_files) == len(fl_ir_files) == len(fr_ir_files)

# each burst has 5 entries. Only keep the first of each burst
fl_rgb_files_to_keep = fl_rgb_files[::5]
fr_rgb_files_to_keep = fr_rgb_files[::5]
fl_ir_files_to_keep = fl_ir_files[::5]
fr_ir_files_to_keep = fr_ir_files[::5]

fl_rgb_files_to_remove = [x for x in fl_rgb_files if x not in fl_rgb_files_to_keep]
fr_rgb_files_to_remove = [x for x in fr_rgb_files if x not in fr_rgb_files_to_keep]
fl_ir_files_to_remove = [x for x in fl_ir_files if x not in fl_ir_files_to_keep]
fr_ir_files_to_remove = [x for x in fr_ir_files if x not in fr_ir_files_to_keep]

for f in fl_rgb_files_to_remove:
    print(f)
    os.remove(f)
for f in fr_rgb_files_to_remove:
    print(f)
    os.remove(f)
for f in fl_ir_files_to_remove:
    print(f)
    os.remove(f)
for f in fr_ir_files_to_remove:
    print(f)
    os.remove(f)

fl_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fl_rgb/*.png')))
fr_rgb_files = sorted(glob(os.path.join(dataroot, '*/*/fr_rgb/*.png')))
fl_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fl_ir/*.png')))
fr_ir_files = sorted(glob(os.path.join(dataroot, '*/*/fr_ir/*.png')))

print(len(fl_rgb_files))
print(len(fr_rgb_files))
print(len(fl_ir_files))
print(len(fr_ir_files))