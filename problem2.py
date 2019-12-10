import config
import scipy.io

def main():
    # alias
    conf = config.C.PROBLEM2

    data = scipy.io.loadmat(conf.DATASET_PATH)['data']
   

if __name__ == '__main__':
    main()
