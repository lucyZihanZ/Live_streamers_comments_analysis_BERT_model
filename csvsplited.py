import pandas as pd

def main(paths,col,num):
    df = pd.read_csv(paths,header=0)  #标题行不包括在内。 如果标题行也要分列的话 改：header=None
    for i in range(num):
        df['pick{}'.format(i + 1)] = df.iloc[:, col].apply(
            lambda x: str(x)[i * len(str(x)) // num:(i + 1) * len(str(x)) // num])
    df.to_csv('"D:\学习\科研助手\新建文件夹\splited9.csv',encoding='gbk')

"""



"""

if __name__ =='__main__':
    paths = 'D:\学习\科研助手\新建文件夹\answer.csv'  #excel文件的路径 如果在该文件夹下，可以直接写文件名，注意不要漏了后缀
    col = 0     #要分第几列 （第一列是0，第二列是1，第三列是2...以此类推）
    num = 5     #要分成几列
    main(paths,col,num)
