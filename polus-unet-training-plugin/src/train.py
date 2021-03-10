
import subprocess
import os
from pathlib import Path


def run_unet_training(modelfile_path, weightfile_path,solverPrototxtAbsolutePath, outDir, gpu_flag='',
                          cleanup=True):

    #fix parameters
    n_inputchannels=1


    command_sanitycheck = []
    command_sanitycheck.append("caffe_unet")
    command_sanitycheck.append("check_model_and_weights_h5")
    command_sanitycheck.append("-model")
    command_sanitycheck.append(modelfile_path)
    command_sanitycheck.append("-weights")
    command_sanitycheck.append(weightfile_path)
    command_sanitycheck.append("-n_channels")
    command_sanitycheck.append(str(n_inputchannels))
    if gpu_flag:
        command_sanitycheck.append("-gpu")
        command_sanitycheck.append(gpu_flag)
    output1 = subprocess.check_output(command_sanitycheck, stderr=subprocess.STDOUT).decode()
    print(output1)
    
    #assemble prediction command
    command_predict = []
    command_predict.append("caffe")
    command_predict.append("train")
    command_predict.append("-solver")
    command_predict.append(solverPrototxtAbsolutePath)
    command_predict.append("-weights")
    command_predict.append(weightfile_path)
    command_predict.append("-gpu")
    command_predict.append(gpu_flag)
    if gpu_flag:
        command_predict.append("-gpu")
        command_predict.append(gpu_flag)
    command_predict.append("-sigint_effect")
    command_predict.append("stop")
    #runs command 
    try:
        output = subprocess.check_output(command_predict, stderr=subprocess.STDOUT).decode()
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output.decode()) # print out the stdout messages up to the exception
        print(e)


    filename = str(Path(outDir)/"results.txt")
    file = open(filename, "w+")
    file.write("w")
    results = output.splitlines()
    for line in results:
        file.write(line+"\n")
    file.close()
    
    os.system("cp snapshot_iter_100.caffemodel.h5 "+outDir)