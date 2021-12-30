import argparse
from glob import glob

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--wav_dir", default="./wav_dirs")
  parser.add_argument("--filelists", default="filelists/train_files_ss.txt")

  args = parser.parse_args()
    
  filenames=glob('{}/**/*.wav'.format(args.wav_dir),recursive=True)
  print("start")
  for filename in filenames:
    with open(args.filelists,"a",encoding="utf-8") as f:
        f.writelines(filename + "\n")
