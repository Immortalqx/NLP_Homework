import re


def read_data(filename=None):
    """
    读取数据文件，并且进行处理，去除标点符号
    :param filename: 文件名，string类型
    :return: 文件的内容
    """
    if filename is None:
        return None
    with open(filename) as file:
        # 首先按行分词
        data_raw = file.read().split("\n")
        data = []
        for line_data in data_raw:
            if line_data is None:  # 去除一些异常情况
                continue
            # 因为标点符号一般不会出现在句首，所以这里用下面的方式刚好可以去除标点符号和多余的空格
            line_data = re.sub(r' [^\w\s]', '', line_data)
            # # 试一试直接用正则表达式【直接使用无法判断句尾空格的情况，会导致后面出问题】
            # line_data = re.sub(r'[^\w\s]', '', line_data)  # 先去除标点符号
            # line_data = re.sub(r'\s+', ' ', line_data)  # 再将多个空格替换为一个空格
            data.append(line_data.split(" "))
        return data


if __name__ == "__main__":
    print(read_data("data/input.txt"))
