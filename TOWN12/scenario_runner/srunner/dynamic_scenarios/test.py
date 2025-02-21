# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 下午8:38
# @Author  : Hcyang
# @File    : test.py
# @Desc    : xxx


import os
import sys
import ipdb
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import math
import carla
from easydict import EasyDict as Edict
from functions import *


def normalize(v):
    # Desc: 将向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def main():
    client = carla.Client('localhost', 2000)
    client.reload_world()
    world = client.get_world()
    wmap = world.get_map()

    wp = wmap.get_waypoint(carla.Location(x=0, y=0, z=0))
    wp = wp.next(70)[0]
    junction = get_next_junction(wp)
    side_walk_wps = get_junction_sidewalk_wps(junction)
    side_walk_wps = sort_wp_by_ref(side_walk_wps, wp)
    pedestrian_start_wp = side_walk_wps[0]
    pedestrian_end_wp = side_walk_wps[1]

    debug = world.debug
    debug.draw_point(pedestrian_start_wp.transform.location + carla.Location(z=0.2), size=0.15, color=carla.Color(255, 0, 0))
    debug.draw_point(pedestrian_end_wp.transform.location + carla.Location(z=0.2), size=0.15, color=carla.Color(0, 255, 0))

    start_forward_vec = pedestrian_start_wp.transform.get_forward_vector()
    end_forward_vec = pedestrian_end_wp.transform.get_forward_vector()
    start_next_loc = pedestrian_start_wp.transform.location + start_forward_vec * 1
    end_next_loc = pedestrian_end_wp.transform.location + end_forward_vec * 1

    debug.draw_point(start_next_loc + carla.Location(z=0.2), size=0.15, color=carla.Color(0, 0, 255))
    debug.draw_point(end_next_loc + carla.Location(z=0.2), size=0.15, color=carla.Color(0, 255, 255))

    tmp = wp.get_right_lane().get_right_lane().get_right_lane().get_right_lane()
    debug.draw_point(tmp.transform.location + carla.Location(z=0.2), size=0.15, color=carla.Color(0, 0, 255))
    # tmp_forward_vec = tmp.transform.get_forward_vector()
    # tmp_next_loc = tmp.transform.location + tmp_forward_vec * 1
    tmp_next_loc = tmp.next_until_lane_end(1)[-1].transform.location
    debug.draw_point(tmp_next_loc + carla.Location(z=0.2), size=0.15, color=carla.Color(255, 0, 0))


    ipdb.set_trace(context=10)
    pause = 1


if __name__ == '__main__':
    main()
