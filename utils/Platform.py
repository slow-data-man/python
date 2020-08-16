import platform


def cur_system():
    if platform.system() == 'Windows':
        return "win"
    elif platform.system() == 'Linux':
        return "linux"
    else:
        return "other"


if __name__ == '__main__':
    cur_sys = cur_system()
    print(cur_system())