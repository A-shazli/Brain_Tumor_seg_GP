import os
import nibabel as nib
import glob

root_path = 'Sample BraTS21 50 Examples'
data_list = sorted(glob.glob(root_path + '/*'))        

def find_tumor(wid, hei):
    slice2D = img.get_fdata()[:, :, i]
    for j in range(wid):
        for k in range(hei):
            if slice2D[j, k] != 0:
                return i
    return 0

results_path_path = r'Sample BraTS21 50 Examples Depth Cropped'
if not os.path.exists(results_path_path):
    os.makedirs(results_path_path)

for vol in data_list:
    modules_list = sorted(glob.glob(vol + '/*'))
    t1_vol_path = modules_list[2]
    img = nib.load(t1_vol_path)
    height, width, depth = img.shape
    filled_slices = []

    for i in range(depth):
        depth_idx = find_tumor(width, height)
        if depth_idx != 0:
            filled_slices.append(depth_idx)

    min_depth_idx, max_depth_idx = filled_slices[0], filled_slices[-1]
    for module_path in modules_list:
        module_vol = nib.load(module_path)
        cropped_vol = module_vol.get_fdata()[:, :, min_depth_idx : (max_depth_idx+1)]
        nifti_img =  nib.Nifti1Image(cropped_vol, module_vol.affine)     # to save this 3D (ndarry) numpy use this

        vol_new_path = results_path_path + '/' +vol.split('\\')[1]
        if not os.path.exists(vol_new_path):
            os.makedirs(vol_new_path)

        module_new_path = vol_new_path + '/' + module_path.split('\\')[-1]
        nib.save(nifti_img, module_new_path)
