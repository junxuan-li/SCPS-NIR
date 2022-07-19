import os
import glob
import yaml
from cfgnode import CfgNode


def same_netconfig_for_diligent_datasets(template_file, new_exp_name, output_yml, data_path, objects):
    with open(template_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    output_yml_list = [os.path.join(output_yml, s + '.yml') for s in objects]
    data_path_list = [os.path.join(data_path, s + 'PNG') for s in objects]
    log_path_list = [os.path.join('./runs', new_exp_name, s) for s in objects]
    for i in range(len(objects)):
        cfg.experiment.log_path = log_path_list[i]
        cfg.dataset.data_path = data_path_list[i]

        with open(output_yml_list[i], 'w') as file:
            documents = cfg.dump(stream=file)


def same_netconfig_for_other_datasets(template_file, new_exp_name, output_yml, data_path, objects):
    with open(template_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    output_yml_list = [os.path.join(output_yml, s + '.yml') for s in objects]
    data_path_list = [os.path.join(data_path, s) for s in objects]
    log_path_list = [os.path.join('./runs', new_exp_name, s) for s in objects]
    for i in range(len(objects)):
        cfg.experiment.log_path = log_path_list[i]
        cfg.dataset.data_path = data_path_list[i]

        with open(output_yml_list[i], 'w') as file:
            documents = cfg.dump(stream=file)


if __name__ == '__main__':
    same_netconfig_for_diligent_datasets(
        template_file='configs/template.yml',
        new_exp_name='diligent',
        output_yml='configs/diligent',
        data_path='data/DiLiGenT/pmsData',
        objects=['ball', 'bear', 'buddha', 'cat', 'cow', 'goblet', 'harvest', 'pot1', 'pot2', 'reading'],
    )

    same_netconfig_for_other_datasets(
        template_file='configs/template.yml',
        new_exp_name='apple',
        output_yml='configs/apple',
        data_path='data/Apple_Dataset',
        objects=['apple', 'gourd1', 'gourd2'],
    )

    print('done')

