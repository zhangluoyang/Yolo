import multiprocessing
from multiprocessing import Pool


def add(n):
    s = 0
    for i in range(n + 1):
        s += i
    return (n, s)


def calculate():
    pool = Pool(processes=4)

    tasks = range(10)
    result_list = list()
    info_dict = dict()

    for n in tasks:
        result_list.append(pool.apply_async(add, (n,)))

    # pool.close()
    # pool.join()

    for result in result_list:
        print("############# type:{0}".format(type(result)))
        k, v = result.get()
        info_dict[k] = v

    return info_dict


def print_result():
    info_dict = calculate()

    key_list = sorted(info_dict.keys())

    for key in key_list:
        print("%s: %s" % (key, info_dict[key]))


if __name__ == '__main__':
    calculate()
    print_result()
