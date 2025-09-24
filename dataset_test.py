import scipy

if __name__ == '__main__':
    # 读取 .mat 文件
    mat = scipy.io.loadmat('./dataset/ACM.mat')

    # 查看文件内容
    print(mat.keys())
