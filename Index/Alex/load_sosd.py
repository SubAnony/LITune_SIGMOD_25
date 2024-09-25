import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os


def load_sosd(file, length):

    data = np.fromfile(file, np.uint64)[1:length+1]

    return data


def plt_sosd(data, name):

    plt.plot(range(len(data)),data)

    plt.xlabel('Index')
    plt.ylabel('Key')
    plt.title('SOSD : ' + name.split('_')[-1])

    plt.grid(True)
    plt.show()

def split_and_save(data, no_seg, output_dir, name):

    splited_array = np.split(data, no_seg)

    # print(splited_array)

    for i,arr in enumerate(splited_array):

        with open (output_dir + name + "_" + str(i),'wb') as f:
        
            f.write(struct.pack("Q", len(arr)))
            arr.astype(np.int64).tofile(f)

def load_and_concate(output_path, file_name, k,file_prefix):

    files_to_load = glob.glob(output_path+file_name+"_*")
    # print(sorted(files_to_load))
    # print(load_lenglth / no_segements)
    concate_arr = np.array([load_sosd(file, int(load_length / no_segements)) for file in sorted(files_to_load)])
    concate_arr = np.concatenate(concate_arr)


    with open (f"./data_SOSD_{file_prefix}/data_%d"%k,'wb') as f:
        
        f.write(struct.pack("Q", len(concate_arr)))
        concate_arr.astype(np.int64).tofile(f)

    
    return concate_arr



if __name__ == "__main__":

    file_prefix = "concat_MIX" #fb,books,OSM,MIX
    file_pre_prefix = 'MIX'
    file_dir = "../data/"
    file_name = f"{file_prefix}_200M_uint64"
    load_length = 180000000
    no_segements = 60
    output_path = "../data/split_data/"

    assert not (load_length % no_segements), "cannot be equally splited"

    sosd = load_sosd(file_dir + file_name, load_length)

    
    # print(sosd)
    # plt_sosd(sosd, file_name)

    os.system("rm -rf ../data/split_data/*")

    split_and_save(sosd, no_segements, output_path, file_name)



    for i in range(30):
        os.mkdir('../data/split_data/%d'%i)
        print("finish creating the %d th data folder"%i)
        for k in range(30):
            os.system(f"cp ../data/split_data/{file_prefix}_200M_uint64_%d ../data/split_data/%d/"%(k+i,i))

    for k in range(30):

        concat_path = "../data/split_data/"+str(k)+'/'
        load_and_concate(concat_path, file_name,k,file_pre_prefix)

    # concate_array = load_and_concate(output_path, file_name)
    # print(concate_array)


