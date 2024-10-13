import os
import soundfile as sf

def convertOGGToWAV(input_folder, output_folder):
    exclude = ["f8c41471412a725a7dd61a5c03a383036b6c17d15ee8096b270e250e8e95718c", "449b4d69ea21060aab89cbf53640107bde977f540b9e1f8be2fa9ec0c8822d89", "fff83d45e9cb441ba3fdbb7ed07676b4aa23d03f3acc4b621dbec4f26a96e45c", 
            "40ea8855a3e1fca6695bad939c5c6311c76860cbe577eade4f0ad599c94fb8ef", "e75c4e774dcb7ac868787bb77108c323683b958bb47cdd39d7eb750c127e36de", "35abb73e93acc5357b1c2fbb151b37ded67c337a72065b9c2f63e0642bbb6084", 
            "d4446b1914c10b5992ac48084aa29dcee2c51e347ff57de76fd2dacd904f4131", "de6fd5edefc695c0482d06b4d3c1364a93c814166049726b323ef812cfc4b76e", "b4ce8c64f7f084c0bc77a4b52d44b82bdef833e61722fa2c82ab4d59e918bc9d",
            "afa1571cf354f7fe13782972a5e10d5d78a22785de42a785803ebea77b58638a", "bc67d6c9efc51c6a3144d21aeda2668324c55943f11135c38130e3527a22188f"]

    i = 0
    for dir in os.listdir(input_folder):
        if(not dir in exclude):
                for root, _, files in os.walk(os.path.join(input_folder, dir)):
                    for name in files:
                        if name.endswith('.ogg'):
                            path = os.path.join(root, name)
                            print(path)
                            data, samplerate = sf.read(path)
                            print(samplerate)
                            exit(0)
                            output_file_path = os.path.join(output_folder, os.path.splitext(name)[0] + '.wav')
                            print(output_file_path)
                            sf.write(output_file_path, data, samplerate)
                    break
                else:
                    continue
                break
        
        
        print(i)

def convertFLACToWAV(input_folder, output_folder):
    for dir in os.listdir(input_folder):
        print(dir)
        for root, _, files in os.walk(os.path.join(input_folder, dir)):
            for name in files:
                if name.endswith('.flac'):
                    path = os.path.join(root, name)
                    data, samplerate = sf.read(path)
                    output_file_path = os.path.join(output_folder, os.path.splitext(name)[0] + '.wav')
                    sf.write(output_file_path, data, 16000)
            #break
        #else:
            #continue
        #break

def createTSVFile(data_dir, train_split):
    train_threshold = int(len(os.listdir(data_dir)) * train_split)
    print(train_threshold)
    train_file_path = os.path.join(data_dir, "train.tsv")
    valid_file_path = os.path.join(data_dir, "valid.tsv")
    f = open(valid_file_path, "w+")
    f.close()
    f = open(train_file_path, "w+")
    f.close()
    f = open(train_file_path, "a")
    f.write(data_dir+"\n")
    for i, filename in enumerate(os.listdir(data_dir)):
        if(filename.endswith(".wav")):
            if(i == train_threshold):
                f.close()
                f = open(valid_file_path, "a")
                f.write(data_dir+"\n")
            data, _ = sf.read(os.path.join(data_dir, filename))
            f.write(filename+"\t"+str(len(data))+"\n")
        else:
            i -= 1
     

if(__name__ == "__main__"):
    convertFLACToWAV("B:\DLL\Datasets\LibriSpeech\\train-clean-100", "B:\DLL\Datasets\LibriSpeech_wav")
    #convertOGGToWAV("B:\DLL\Datasets\downloads\extracted", "B:\DLL\Datasets\XCM_wav")
    #createTSVFile("B:\DLL\Datasets\XCM_wav", 0.8)