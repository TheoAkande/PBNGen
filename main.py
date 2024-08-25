from Generator import HSV, PBNImage, Mode


def try_input(message):
    while True:
        try:
            val = int(input(message))
            return val
        except ValueError:
            print("Invalid input, try again")


if __name__ == '__main__':
    test_image = r"C:\Users\theoa\Pictures\test_image.jpg"
    pycharm = r"C:\Users\theoa\Pictures\pycharm-PyCharm_400x400_Twitter_logo_white.png"
    earth = r"C:\Users\theoa\Pictures\globe-700x700.jpg"
    mode = Mode.MYMETHOD

    name = input("File path: ")
    if name == "":
        name = pycharm
    num = try_input("Enter number of colours: ")
    rgb = try_input("Enter the rgb depth: ")
    gen = PBNImage(name, num, rgb)
    gen.do_all(mode)


