import os
import sys
import tacorn.fileutils as fu
import config
import wavernn.train


def main():
    # TODO: get from cmd line
    pretrained_model = "LJ"
    workdir = "workdir"
    cfg = config.Configuration()

    print("This script is not functional yet, please use train.sh")
    sys.exit(1)

    cfg.paths = create_workdir_structure(workdir)

    # TODO: preprocessing

    if pretrained_model:
        cfg.paths["wavernn_pretrained_model"] = get_pretrained_wavernn(
            pretrained_model, cfg.paths["wavernn_pretrained"])

    wavernn.train.train(cfg)


if __name__ == '__main__':
    main()
