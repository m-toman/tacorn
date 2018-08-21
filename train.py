import os
import tacorn.fileutils as fu
import config
import wavernn.train


pretrained_wavernn_models = {
    "LJ": ("https://www.dropbox.com/s/raw/bwgrvvmxk9y8fov/lj.pyt", "lj.pyt")
}


def get_pretrained_wavernn(model_id, targetdir):
    """ Downloads a pretrained waveRNN model. """
    if model_id not in pretrained_wavernn_models:
        raise FileNotFoundError(model_id)

    (model_url, model_filename) = pretrained_wavernn_models[model_id]
    model_dir = os.path.join(targetdir, model_id)
    model_path = os.path.join(model_dir, model_filename)
    if os.path.exists(model_path):
        return

    fu.ensure_dir(model_dir)
    fu.download_file(model_url, model_path)
    return model_path


def create_workdir_subfolder(workdir_path, folder):
    absfolder = os.path.join(workdir_path, folder)
    fu.ensure_dir(absfolder)
    return absfolder


def create_workdir_structure(workdir_path):
    fu.ensure_dir(workdir_path)
    # TODO: make paths an object that throws an exception when dirname does not exist
    paths = {}
    paths["input_data"] = create_workdir_subfolder(workdir_path, "input_data")
    paths["wavernn_pretrained"] = create_workdir_subfolder(
        workdir_path, "wavernn_pretrained")
    return paths


def main():
    # TODO: get from cmd line
    pretrained_model = "LJ"
    workdir = "workdir"
    cfg = config.Configuration()
    
    cfg.paths = create_workdir_structure(workdir)

    # TODO: preprocessing

    if pretrained_model:
        cfg.paths["wavernn_pretrained_model"] = get_pretrained_wavernn(
            pretrained_model, cfg.paths["wavernn_pretrained"])

    wavernn.train.train(cfg)


if __name__ == '__main__':
    main()
