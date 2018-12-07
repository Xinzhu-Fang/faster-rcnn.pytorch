import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import math
import random
import numpy as np
import os
import pandas as pd
from itertools import permutations
import pdb
import shutil



class PopoutImages(object):
    def __init__(self):
        self.num_images = 1#0 # gratings 6*8 cond; circles 6* 60 cond
        self.on_grid = False # clutter
        self.fixed_size = True # size of each item
        self.num_item = range(3, 9) # 3 to 10
        self.type = [0, 1] # color and grating
        self.dir = 'images'
        self.image_width = 600 #512
        self.image_height = 600 #512
        print(self.image_width)
        self.num_unit = 10
        self.unit_width = self.image_width / self.num_unit #60
        self.width_units = math.floor(self.image_width / self.unit_width)
        self.height_units = math.floor(self.image_height / self.unit_width)
        if os.path.isdir(os.path.join(self.dir)):
            pass
        else:
            os.mkdir(self.dir)


class PopoutCircles(PopoutImages):
    def __init__(self):
        PopoutImages.__init__(self)
        self.item_radius = self.unit_width/3 #20
        self.sub_dir = 'circles'
        self.color_avail = np.array([[0, 0, 0],
                                    [255, 255, 255],
                                    [255, 0, 0],
                                    [0, 255, 0],
                                    [0, 0, 255]])
        # 0 background, 1 distractor, 2 target
        self.color_choices = list(permutations(range(self.color_avail.shape[0]), 3))
        # self.col_names = ['image_file_name', 'item_radius', 'target_x', 'target_y', 'num_item', 'target_color',
        #                   'back_color', 'distractor_color']

    # def create_single(self):


    def create(self):
        if os.path.isdir(os.path.join(self.dir, self.sub_dir)):
            shutil.rmtree(os.path.join(self.dir, self.sub_dir))
        os.mkdir(os.path.join(self.dir, self.sub_dir))
        df_image_labels = pd.DataFrame()
        # df_image_labels = pd.DataFrame(columns=self.col_names)

        image_ind = -1
        for iCC in range(len(self.color_choices)):
            radius = self.item_radius
            back_color = self.color_avail[self.color_choices[iCC][0], :]
            distractor_color = self.color_avail[self.color_choices[iCC][1], :]
            target_color = self.color_avail[self.color_choices[iCC][2], ]
            for iNum in self.num_item:
                for iImage in range(self.num_images):
                    image_ind += 1
                    x_positions = np.array(random.sample(range(self.width_units), iNum)) * self.unit_width + self.unit_width/2
                    y_positions = np.array(random.sample(range(self.height_units), iNum)) * self.unit_width + self.unit_width/2
                    # radius = np.repeat(self.item_radius, iNum)
                    # radius = np.array(random.sample(range(min_radius * 10, self.unit_width * 10), iNum)) / 10
                    # target_height = random.randint(0, self.height_units-1)
                    # target_width = random.randint(0, self.width_units-1)
                    target_ind = random.randint(0, iNum-1)
                    all_colors = np.repeat(distractor_color[np.newaxis, :], iNum, axis=0)
                    all_colors[target_ind, :] = target_color

                    image = Image.new("RGB", (self.image_width, self.image_height))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([0, 0, self.image_width, self.image_height], fill=tuple(back_color.reshape(1, -1)[0]))

                    for iItem in range(iNum):
                        x0 = x_positions[iItem] - radius#[iItem]
                        x1 = x_positions[iItem] + radius#[iItem]
                        y0 = y_positions[iItem] - radius#[iItem]
                        y1 = y_positions[iItem] + radius#[iItem]
                        color = all_colors[iItem]
                        # pdb.set_trace()
                        # draw = ImageDraw.Draw(image)
                        draw.ellipse([x0, y0, x1, y1], fill=tuple(color.reshape(1, -1)[0]))
                    image_file_name = '%04d' % image_ind + '.png'
                    image.save(os.path.join(os.getcwd(), self.dir, self.sub_dir, image_file_name))

                    df_image_labels = df_image_labels.append({'image_file_name': image_file_name, 'item_radius': radius,
                                                              'target_x': x_positions[target_ind],
                                                              'target_y': y_positions[target_ind],
                                                              'num_item': iNum, 'target_color_r': target_color[0], 'target_color_g': target_color[1],
                                                              'target_color_b': target_color[2], 'back_color_r': back_color[0], 'back_color_g': back_color[1],
                                                              'back_color_b': back_color[2], 'distractor_color_r': distractor_color[0],
                                                              'distractor_color_g': distractor_color[1], 'distractor_color_b': distractor_color[2]},
                                                             ignore_index=True)

                    df_image_labels.to_csv(os.path.join(self.dir, 'df_' + self.sub_dir + '.csv'), encoding='utf-8')


class PopoutGratings(PopoutImages):
    def __init__(self):
        PopoutImages.__init__(self)
        self.sub_dir = 'gratings'
        self.distractor_angles = np.array(range(8)) * (1/8) * math.pi
        self.target_angles = self.distractor_angles + 1/2 * math.pi
        self.grating_color = (200, 200, 200)
        self.item_radius = self.unit_width/3 #math.sqrt(self.grating_height ** 2 + self.grating_width ** 2)
        self.grating_angle = 1/15 * math.pi #math.atan(self.grating_width/self.grating_height)
        self.grating_height = self.item_radius * math.sin(self.grating_angle) #20 #half of the height
        self.grating_width = self.item_radius * math.cos(self.grating_angle) #3
        # self.col_names = ['image_file_name', 'item_radius', 'target_x', 'target_y', 'num_item', 'target_angle']

    def create(self):
        if os.path.isdir(os.path.join(self.dir, self.sub_dir)):
            shutil.rmtree(os.path.join(self.dir, self.sub_dir))
        os.mkdir(os.path.join(self.dir, self.sub_dir))
        # df_image_labels = pd.DataFrame(columns=self.col_names)
        df_image_labels = pd.DataFrame()

        image_ind = -1
        for iA in range(len(self.distractor_angles)):
            radius = self.item_radius
            distractor_angle= self.distractor_angles[iA]
            target_angle= self.target_angles[iA]
            distractor_xs = np.array([math.cos(distractor_angle + self.grating_angle),
                                      math.cos(distractor_angle - self.grating_angle),
                                      math.cos(distractor_angle + self.grating_angle + math.pi),
                                      math.cos(distractor_angle - self.grating_angle + math.pi)]) * radius
            distractor_ys = np.array([math.sin(distractor_angle + self.grating_angle),
                                      math.sin(distractor_angle - self.grating_angle),
                                      math.sin(distractor_angle + self.grating_angle + math.pi),
                                      math.sin(distractor_angle - self.grating_angle + math.pi)]) * radius * (-1)
            target_xs = np.array([math.cos(target_angle + self.grating_angle),
                                  math.cos(target_angle - self.grating_angle),
                                  math.cos(target_angle + self.grating_angle + math.pi),
                                  math.cos(target_angle - self.grating_angle + math.pi)]) * radius
            target_ys = np.array([math.sin(target_angle + self.grating_angle),
                                  math.sin(target_angle - self.grating_angle),
                                  math.sin(target_angle + self.grating_angle + math.pi),
                                  math.sin(target_angle - self.grating_angle + math.pi)]) * radius * (-1)
            for iNum in self.num_item:
                for iImage in range(self.num_images):
                    image_ind += 1
                    x_positions = np.array(random.sample(range(self.width_units), iNum)) * self.unit_width + self.unit_width/2
                    y_positions = np.array(random.sample(range(self.height_units), iNum)) * self.unit_width + self.unit_width/2

                    target_ind = random.randint(0, iNum-1)
                    all_xs = np.repeat(distractor_xs[np.newaxis, :], iNum, axis=0)
                    all_ys = np.repeat(distractor_ys[np.newaxis, :], iNum, axis=0)
                    all_xs[target_ind, :] = target_xs
                    all_ys[target_ind, :] = target_ys
                    all_xs = (all_xs.T + x_positions).T
                    all_ys = (all_ys.T + y_positions).T

                    image = Image.new("RGB", (self.image_width, self.image_height))
                    draw = ImageDraw.Draw(image)

                    for iItem in range(iNum):
                        # pdb.set_trace()
                        # draw = ImageDraw.Draw(image)
                        coor = []
                        for iCorner in range(4):
                            coor.append(all_xs[iItem][iCorner])
                            coor.append(all_ys[iItem][iCorner])
                        draw.polygon(coor, fill=self.grating_color)
                    image_file_name = '%04d' % image_ind + '.png'
                    image.save(os.path.join(os.getcwd(), self.dir, self.sub_dir,  image_file_name))

                    df_image_labels = df_image_labels.append({'image_file_name': image_file_name, 'item_radius': radius,
                                                              'target_x': x_positions[target_ind], 'target_y': y_positions[target_ind],
                                                              'num_item': iNum, 'target_angle': target_angle}, ignore_index=True)
                    df_image_labels.to_csv(os.path.join(self.dir, 'df_' + self.sub_dir + '.csv'), encoding='utf-8')

