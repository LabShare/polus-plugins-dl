
import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger("training")
logger.setLevel(logging.INFO)


def run_unet_training(modelfile_path, weightfile_path,solverPrototxtAbsolutePath, outDir, gpu_flag='',
                          cleanup=True):
    """Run unet training.

    Run unet training using caffe binary.

    Args:
        modelfile_path: model file path
        weightfile_path: weights file path
        solverPrototxtAbsolutePath: solver prototxt file
        outDir: output directory

    """

    #assemble prediction command
    command_predict = []
    command_predict.append("caffe")
    command_predict.append("train")
    command_predict.append("-solver")
    command_predict.append(solverPrototxtAbsolutePath)
    command_predict.append("-weights")
    command_predict.append(weightfile_path)
    if gpu_flag:
        command_predict.append("-gpu")
        command_predict.append(gpu_flag)
    command_predict.append("-sigint_effect")
    command_predict.append("stop")
    #runs command
    output = subprocess.check_output(command_predict, stderr=subprocess.STDOUT).decode()
    logger.info("training output = {}".format(output))
    filename = str(Path(outDir)/"results.txt")
    file = open(filename, "w+")
    file.write("w")
    results = output.splitlines()
    for line in results:
        file.write(line+"\n")
    file.close()