# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 下午8:35
# @Author  : Hcyang
# @File    : tiny_scenarios.py
# @Desc    : xxx

import re
import random
import carla
import numpy as np
from random import choice
import os
import argparse
import pickle
import json
from .tiny_scenarios import VEHICLE_TYPE_DICT, TYPE_VEHICLE_DICT, choose_bp_name
from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from functools import reduce
# import ipdb


# Special: 参数说明
# - wp: 参考waypoint
# - actor_list: 生成的actor列表
# - actor_desc: 生成的actor描述
# - scene_cfg: 生成车辆的配置
#   - filters: actor蓝图的过滤条件           | '+common'
#   - idp: 生成车辆的概率                    | 0.5
#   - lane_num: 生成车辆的车道数             | 999
#   - self_lane: 是否生成在当前车道上        | False
#   - forward_num: 向前生成车辆的数量         | 6
#   - backward_num: 向后生成车辆的数量        | 4
# - gen_cfg: 生成车辆的配置
#   - name_prefix: 生成车辆的前缀            | 'vehicle'


def _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, name_prefix='vehicle'):
    offset_index = 0
    for v_index, (v_bp, v_transform) in enumerate(bp_and_transforms):
        right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform, retry=1)
        if right_actor is not None:
            actor_list.append(right_actor)
            actor_desc.append('_'.join([name_prefix, str(v_index-offset_index)]))
        else:
            offset_index += 1


def _warning_unused_keys(kwargs):
    for k in kwargs:
        print(f'Warning: Unused key {k} in kwargs')


def _traffic_flow_scenario(wp, filters='+common', idp=0.5, forward_num=6, backward_num=4, **kwargs):
    # Desc: 在当前waypoint的左侧车道或者右侧车道生成车流
    results = []

    # Desc: 先向前生成车流
    _vehicle_wp = wp
    right_forward_index = 1
    while right_forward_index <= forward_num:
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp.transform))
        _vehicle_wps = _vehicle_wp.next(random.randint(8, 15))
        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        right_forward_index += 1

    # Desc: 再向后生成车流
    _vehicle_wp = wp
    right_backward_index = 1
    while right_backward_index <= backward_num:
        _vehicle_wps = _vehicle_wp.previous(8)
        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp.transform))
        right_backward_index += 1

    return results


def right_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成交通流，如果右侧为行驶车道
    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.append(wp.lane_id)

    driving_lane_count = 0
    while wp is not None:
        wp = wp.get_right_lane()
        if wp is None:
            return
        if reduce(lambda x, y: x * y, [wp.lane_id, processed_lanes[0]]) < 0:
            break
        if wp.lane_type != carla.LaneType.Driving or wp.lane_id in processed_lanes or driving_lane_count >= scene_cfg.get('lane_num', 999):
            return
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1


def left_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的左侧车道生成交通流，如果左侧为行驶车道
    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.append(wp.lane_id)

    driving_lane_count = 0
    while wp is not None:
        wp = wp.get_left_lane()
        if wp is None:
            return
        if reduce(lambda x, y: x * y, [wp.lane_id, processed_lanes[0]]) < 0:
            break
        if wp.lane_type != carla.LaneType.Driving or wp.lane_id in processed_lanes or driving_lane_count >= scene_cfg.get('lane_num', 999):
            break
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1


def opposite_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前道路的对向车道生成交通流

    # Special: 获取当前车道的对向车道的最左侧的waypoint
    added_lanes = []
    last_wp = None
    while True:
        if wp is None:
            return
        if wp.lane_id in added_lanes:
            break
        added_lanes.append(wp.lane_id)
        last_wp = wp
        wp = wp.get_left_lane()

    if last_wp is None:
        return

    while last_wp.lane_type != carla.LaneType.Driving:
        if last_wp is None:
            return
        last_wp = last_wp.get_right_lane()

    scene_cfg.update({'self_lane': True})
    right_traffic_flow_scenario(last_wp, actor_list, actor_desc, scene_cfg, gen_cfg)


def right_parking_vehicle_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成停车车辆
    processed_lanes = set()
    if scene_cfg.get('self_lane', False):
        if wp.lane_type == carla.LaneType.Stop or (wp.lane_type == carla.LaneType.Shoulder and wp.lane_width >= 2):
            bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
            _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.add(wp.lane_id)

    stop_lane_count = 0
    while True:
        wp = wp.get_right_lane()
        if wp is None:
            return
        if wp.lane_type != carla.LaneType.Stop and (wp.lane_type != carla.LaneType.Shoulder or wp.lane_width < 2):
            continue
        if wp.lane_id in processed_lanes or stop_lane_count >= scene_cfg.get('lane_num', 999):
            return
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.add(wp.lane_id)
