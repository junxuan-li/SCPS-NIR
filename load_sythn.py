import ups_sythn.ups_synth_test_dataset as UPS

"""
Normal coordinates for DiLiGenT dataset: (same for light direction)
    y    
    |   
    |  
    | 
     --------->   x
   /
  /
 z 
x direction is looking right
y direction is looking up
z direction is looking outside the image

we convert it to :
Normal coordinates for DiLiGenT dataset: (same for light direction)
     --------->   x
    |   
    |  
    | 
    y    
x direction is looking right
y direction is looking down
z direction is looking into the image

"""

def load_sythn(syn_obj, light_index, material_idx):
    d = UPS.UpsSynthTestDataset(syn_obj, light_index).__getitem__(material_idx)
    light_dir = d['dirs']
    light_dir[..., 1:] = -light_dir[..., 1:]  # convert y-> -y   z->-z

    light_intensity = d['ints']

    gt_normal = d['normal'].swapaxes(0, 1).swapaxes(1, 2)
    gt_normal[..., 1:] = -gt_normal[..., 1:]  # convert y-> -y   z->-z

    images = d['img'].swapaxes(1, 2).swapaxes(2, 3)
    mask = d['mask'][0]

    out_dict = {'images': images, 'mask': mask, 'light_direction': light_dir, 'light_intensity': light_intensity, 'gt_normal': gt_normal}
    return out_dict
