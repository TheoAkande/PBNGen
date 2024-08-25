import numpy
from PIL import Image, ImageDraw, ImageColor
import colorsys
import numpy as np
import time
import os
import random
from enum import Enum


class Mode(Enum):
    PCA = 1
    KMEANS = 2
    MYMETHOD = 3


def tup_diff(tup1, tup2):
    t = 0
    for i in range(len(tup1)):
        t += tup1[i] - tup2[i]
    return t


def min_2d(list1, ind):
    m = list1[0]
    for x in list1:
        if x[ind] < m[ind]:
            m = x
    return m


def max_2d(list1, ind):
    m = list1[0]
    for x in list1:
        if x[ind] > m[ind]:
            m = x
    return m


class HSV:

    @staticmethod
    def make_hsv(hsv, depth):
        return round(hsv[0] * 2 * np.pi, depth) + 1000 * round(hsv[1], 2) + 10000 * round((hsv[2] / 255) * 100, 0)

    @staticmethod
    def get_h(num):
        wn = num / 10
        return (wn - np.floor(wn)) * 10

    @staticmethod
    def get_s(num):
        return (np.floor(num / 10) % 1000) / 100

    @staticmethod
    def get_v(num):
        return np.floor(num / 10000) / 100

    @staticmethod
    def show(num):
        print(f"H: {HSV.get_h(num) / (2 * np.pi)}, S: {HSV.get_s(num)}, V: {HSV.get_v(num) * 255}")

    @staticmethod
    def get(num):
        return [HSV.get_h(num) / (2 * np.pi), HSV.get_s(num), HSV.get_v(num) * 255]

    @staticmethod
    def make_pdn_friendly(num):
        return [(HSV.get_h(num) / (2 * np.pi)) * 360, HSV.get_s(num) * 100, HSV.get_v(num) * 100]

    @staticmethod
    def distance(c1, c2):
        h1 = HSV.get_h(c1)
        h2 = HSV.get_h(c2)
        s1 = HSV.get_s(c1)
        s2 = HSV.get_s(c2)
        v1 = HSV.get_v(c1)
        v2 = HSV.get_v(c2)
        return (np.sin(h1) * s1 * v1 - np.sin(h2) * s2 * v2) ** 2 + (np.cos(h1) * s1 * v1 - np.cos(h2) * s2 * v2) ** 2 + (v1 - v2) ** 2


class Cluster:
    def __init__(self, first):
        self.items = np.array([])
        self.newItems = np.array([])
        self.mean = first

    def add(self, item):
        self.items = numpy.append(self.items, item)

    def new_add(self, item):
        self.newItems = numpy.append(self.newItems, item)

    def commit(self):
        self.items = self.newItems
        self.update_mean()
        self.newItems = np.array([])

    def update_mean(self):
        if len(self.items) == 0:
            self.mean = 0
        else:
            self.mean = np.mean(self.items)

    def get_mean(self):
        return self.mean


class PBNImage:

    blank_colour = (225, 225, 225)
    line_colour = (175, 175, 175)

    def __init__(self, name=r"C:\Users\theoa\Pictures\test_image.jpg", num_colours=10, rgb_depth=5):
        timer = time.time()
        self.name = name
        self.num_colours = num_colours
        self.depth = 4
        self.valid = True
        self.rgb_depth = 2**rgb_depth
        try:
            self.image_RGB = Image.open(self.name)
            print(f"Image Loaded in {time.time() - timer}s")
            timer = time.time()
        except FileNotFoundError:
            print("Invalid file name")
            self.valid = False
        if self.valid:
            self.pixel_array = list(self.image_RGB.getdata())
            print(f"Created pixel array in {time.time() - timer}s")
            timer = time.time()
            temparr = []
            for c in self.pixel_array:
                x = [int((int((i / 255) * self.rgb_depth) / self.rgb_depth) * 255) for i in c]
                temparr.append(colorsys.rgb_to_hsv(x[0], x[1], x[2]))
            print(f"Converted to HSV in {time.time() - timer}s")
            timer = time.time()
            self.np_pixel_array = np.array(temparr)
            print(f"Converted to np array in {time.time() - timer}s")
            timer = time.time()
            self.width = self.image_RGB.size[0]
            self.height = self.image_RGB.size[1]
            self.colours = {}
            self.unique_colours = 0
            self.reduced_image = []
            for hsv in self.np_pixel_array:
                c = HSV.make_hsv(hsv, self.depth)
                self.reduced_image.append(c)
                try:
                    self.colours[c] += 1
                except KeyError:
                    self.colours[c] = 1
                    self.unique_colours += 1
            print(f"Generated colours dict in {time.time() - timer}s")
            print(f"{self.unique_colours} unique colours")
            self.final_colours = []
            self.weighted_colours = []
            self.matrix = []
            self.coloured_image = []
            self.palette = None
            self.output_image = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
            self.blank_image = Image.new("RGB", (self.width, self.height), color=PBNImage.blank_colour)

    def k_means_colours(self):
        # Randomly produce n clusters
        ws = list(map(lambda x: x[0], sorted(list(self.colours.items()), key=lambda x: x[1], reverse=True)))
        clusters = [Cluster(ws[j]) for j in range(self.num_colours)]
        for c in self.reduced_image:
            clusters[random.randint(0, self.num_colours - 1)].add(c)

        for cl in clusters:
            print(f"Cluster mean: {cl.get_mean()}, size: {len(cl.items)}")
        # Use 10 epochs to find the best clusters
        epochs = 3
        for i in range(epochs):
            print(f"epoch {i}")
            for c in self.reduced_image:
                min_dist = 100000000000000
                min_cluster = None
                for cl in clusters:
                    dist = HSV.distance(cl.get_mean(), c)
                    if dist < min_dist:
                        min_dist = dist
                        min_cluster = cl
                if min_cluster is not None:
                    min_cluster.new_add(c)
            for cl in clusters:
                cl.commit()
                print(f"Cluster mean: {cl.get_mean()}, size: {len(cl.items)}")
        for cl in clusters:
            if cl is not None:
                self.final_colours.append(HSV.get(cl.get_mean()))
                print(f"Cluster mean: {cl.get_mean()}, size: {len(cl.items)}")

    def generate_colours(self):
        timer = time.time()
        min_list = [0, 0]
        self.weighted_colours = [min_list]
        for colour, n in self.colours.items():
            s = self.sum_colours(colour)
            if s > min_list[1]:
                if len(self.weighted_colours) >= self.num_colours:
                    self.weighted_colours.remove(min_list)
                self.weighted_colours.append([colour, s])
                min_list = min_2d(self.weighted_colours, 1)
        print(f"Generated colours in {time.time() - timer}s")
        for l in self.weighted_colours:
            self.final_colours.append(HSV.get(l[0]))

    def generate_colours_pca(self):
        timer = time.time()
        self.matrix = np.full((self.unique_colours, self.unique_colours), 0)
        i = 0
        colours_in_order = []
        for colour, n in self.colours.items():
            colours_in_order.append(colour)
            self.matrix[i] = self.compare_colours(colour)
            i += 1
        print(f"Generated matrix in {time.time() - timer}s")
        timer = time.time()
        std_mat = (self.matrix - self.matrix.mean(axis=0)) / self.matrix.std(axis=0)
        print(f"Standardized matrix in {time.time() - timer}s")
        timer = time.time()
        cov_mat = np.cov(std_mat, ddof=0, rowvar=False)
        print(f"Calculated cov matrix in {time.time() - timer}s")
        timer = time.time()
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        print(f"Calculated eigenvalues in {time.time() - timer}s")
        timer = time.time()
        order = np.argsort(eigenvalues)[::-1]
        # sorted_eigenvalues = eigenvalues[order]
        sorted_eigenvectors = eigenvectors[:, order]
        print(f"Sorted eigenvalues in {time.time() - timer}s")
        timer = time.time()
        # eigen_sum = np.sum(sorted_eigenvalues)
        # explained_variance = sorted_eigenvalues / eigen_sum
        reduced_data = np.matmul(std_mat, sorted_eigenvectors[:, :self.num_colours])
        print(f"Calculate PCA in {time.time() - timer}s")
        timer = time.time()
        max_list = [0, 10]
        cols = [max_list]
        r = 0
        for row in reduced_data:
            s = 0
            for v in row:
                s += abs(v)
            if s < max_list[1]:
                cols.append([r, s])
                if len(cols) > self.num_colours:
                    cols.remove(max_list)
                max_list = max_2d(cols, 1)
            r += 1
        self.weighted_colours = []
        for i in cols:
            self.weighted_colours.append([colours_in_order[i[0]], i[1]])
        print(f"Generated colours in {time.time() - timer}s")
        for l in self.weighted_colours:
            self.final_colours.append(HSV.get(l[0]))

    def compare_colours(self, col):
        arr = []
        for colour, num in self.colours.items():
            if col == colour:
                arr.append(num)
            else:
                arr.append(1 - min(HSV.distance(col, colour), 1))
        return arr

    def sum_colours(self, col):
        t = 0
        for colour, num in self.colours.items():
            if col == colour:
                t += num
            else:
                t += (HSV.distance(col, colour)**3) / 2000
        return t

    def make_palette(self):

        # Create the image
        square_width = 100
        line_width = 3

        image_size = ((square_width + line_width) * self.num_colours - line_width, square_width)  # Width, Height of the resulting image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        # Draw squares for each HSV value
        i = 0
        for hsv in self.final_colours:
            x_start = i * (square_width + line_width)
            y_start = 0
            x_end = x_start + square_width
            y_end = y_start + square_width
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            nrgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            draw.rectangle(((x_start, y_start), (x_end, y_end)), fill=nrgb)
            i += 1
            if i < len(self.final_colours):
                draw.rectangle(((x_start + square_width, y_start), (x_end + 5, y_end)), fill=(0, 0, 0))
        self.palette = image

    def generate_example(self):
        timer = time.time()
        if len(self.final_colours) == 0:
            print("Generate palette first!")
            return
        self.coloured_image = []
        x = 0
        y = 0
        for p in self.reduced_image:
            hsv = HSV.get(self.min_distance(p))
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            nrgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self.output_image.putpixel((x, y), nrgb)
            x += 1
            if x == self.width:
                x = 0
                y += 1
        print(f"Coloured image in {time.time() - timer}s")

    def min_distance(self, colour):
        m = 10000
        col = 0
        for x in self.weighted_colours:
            d = HSV.distance(x[0], colour)
            if d < m:
                m = d
                col = x[0]

        return col

    def get_neighbours_difference(self, point):
        x = point[0]
        y = point[1]
        s = 0
        tup = self.output_image.getpixel(point)
        for i in range(-1, 2):
            for j in range(-1, 2):
                v = x + i
                r = y + j
                if (0 <= v < self.width) and (0 <= r < self.height):
                    s += tup_diff(tup, self.output_image.getpixel((v, r))) ** 2
        return s

    def make_blank(self):
        timer = time.time()
        for i in range(self.width):
            for j in range(self.height):
                if self.get_neighbours_difference((i, j)) != 0:
                    self.blank_image.putpixel((i, j), PBNImage.line_colour)
        print(f"Created blank in {time.time() - timer}s")

    def save(self, name="default"):
        try:
            os.makedirs(name)
        except FileExistsError:
            print("Directory exists - overwriting")
        self.image_RGB.save(f"{name}\\original.png")
        self.palette.save(f"{name}\\palette.png")
        self.output_image.save(f"{name}\\image.png")
        self.blank_image.save(f"{name}\\blank.png")
        print("PBN Saved")

    def do_all(self, mode=Mode.MYMETHOD, name="default"):
        print("Generating colours")
        match mode:
            case Mode.PCA:
                self.generate_colours_pca()
            case Mode.KMEANS:
                self.k_means_colours()
            case Mode.MYMETHOD:
                self.generate_colours()
        print("Generating palette")
        self.make_palette()
        print("Generating example")
        self.generate_example()
        print("Generating blank")
        self.make_blank()
        print("Done")
        self.save(name)
