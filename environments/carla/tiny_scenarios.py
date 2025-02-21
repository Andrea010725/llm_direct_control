import random
import re
from functools import reduce
import sys
# 111 换路径
#sys.path.append('/home/ubuntu/WorkSpacesPnCGroup/czw/carla/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
sys.path.append("/root/czw_carla/My_recent_research/carla-driving-rl-agent-master/carla-0.9.10-py3.7-linux-x86_64.egg")
import carla
import ipdb
import math
import numpy as np
from random import choice
from typing import List,Tuple
# 111 换路径
#sys.path.append("/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/rl/environments/carla")
sys.path.append("/root/czw_carla/para-direct-produced_new/rl/environments/carla")
from carladateprovider import CarlaDataProvider


VEHICLE_TYPE_DICT = {
    'vehicle.audi.a2': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.etron': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.tt': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.bmw.grandtourer': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.chevrolet.impala': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.citroen.c3': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.dodge.charger_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.dodge.charger_police': ['car', 'special', 'police', 'wheel4'],
    'vehicle.dodge.charger_police_2020': ['car', 'special', 'police', 'wheel4'],
    'vehicle.ford.crown': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.ford.mustang': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.jeep.wrangler_rubicon': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2017': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.mercedes.coupe': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.mercedes.coupe_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.micro.microlino': ['car', 'small', 'wheel4'],
    'vehicle.mini.cooper_s': ['car', 'wheel4', 'common'],
    'vehicle.mini.cooper_s_2021': ['car', 'wheel4', 'common'],
    'vehicle.nissan.micra': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol_2021': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.seat.leon': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.toyota.prius': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.carlacola': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.firetruck': ['truck', 'special', 'fire', 'large', 'wheel4'],
    'vehicle.tesla.cybertruck': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.ford.ambulance': ['van', 'special', 'ambulance', 'large', 'wheel4'],
    'vehicle.mercedes.sprinter': ['van', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.volkswagen.t2': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.volkswagen.t2_2021': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.mitsubishi.fusorosa': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.harley-davidson.low_rider': ['moto', 'wheel2', 'common'],
    'vehicle.kawasaki.ninja': ['moto', 'wheel2', 'common'],
    'vehicle.vespa.zx125': ['electric', 'wheel2'],
    'vehicle.yamaha.yzf': ['moto', 'wheel2', 'common'],
    'vehicle.bh.crossbike': ['bicycle', 'wheel2'],
    'vehicle.diamondback.century': ['bicycle', 'wheel2'],
    'vehicle.gazelle.omafiets': ['bicycle', 'wheel2'],
}
TYPE_VEHICLE_DICT = {}
for bp_name_outside, bp_filters_outside in VEHICLE_TYPE_DICT.items():
    for bp_filter_outside in bp_filters_outside:
        if bp_filter_outside not in TYPE_VEHICLE_DICT:
            TYPE_VEHICLE_DICT[bp_filter_outside] = []
        TYPE_VEHICLE_DICT[bp_filter_outside].append(bp_name_outside)


OBSTACLE_TYPE_DICT = {
    # traffic obstacles
    'static.prop.garbage01': ['garbage'],  # 建筑垃圾
    'static.prop.garbage02': ['garbage'],
    'static.prop.garbage03': ['garbage'],
    'static.prop.garbage04': ['garbage'],
    'static.prop.busstop ': ['bus_stop'],   # 公交车站
    'static.prop.constructioncone': ['construction'],   # 施工锥，用于标记施工区域或指引行人和车辆
    'static.prop.streetbarrier': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.warningconstruction': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.trafficcone01': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.trafficcone02': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.warningaccident' :['accident'],
    'walker.pedestrian.0004': ['workers'],
    'walker.pedestrian.0003': ['workers'],
    'walker.pedestrian.0015': ['workers'],
    'walker.pedestrian.0019': ['workers'],
    'walker.pedestrian.0016': ['workers'],
    'walker.pedestrian.0023': ['workers'],
    'static.prop.creasedbox02': ['creasedbox'],
    'static.prop.ironplank': ['plank']
}
TYPE_OBSTACLE_DICT = {}
for bp_obstacle_name, bp_obstacle_filters in OBSTACLE_TYPE_DICT.items():
    for bp_obstacle_filters in bp_obstacle_filters:
        if bp_obstacle_filters not in TYPE_OBSTACLE_DICT:
            TYPE_OBSTACLE_DICT[bp_obstacle_filters] = []
        TYPE_OBSTACLE_DICT[bp_obstacle_filters].append(bp_obstacle_name)

WALKER_TYPE_DICT = {
        'walker.pedestrian.0004': ['workers'],
        'walker.pedestrian.0003': ['workers'],
        'walker.pedestrian.0015': ['workers'],
        'walker.pedestrian.0019': ['workers'],
        'walker.pedestrian.0016': ['workers'],
        'walker.pedestrian.0023': ['workers'],
    }
TYPE_WALKER_DICT = {}
for bp_WALKER_name, bp_WALKER_filters in WALKER_TYPE_DICT.items():
    for bp_WALKER_filters in bp_WALKER_filters:
        if bp_WALKER_filters not in TYPE_WALKER_DICT:
            TYPE_WALKER_DICT[bp_WALKER_filters] = []
        TYPE_WALKER_DICT[bp_WALKER_filters].append(bp_WALKER_name)

SUV = ['vehicle.audi.etron',
       'vehicle.nissan.patrol',
       'vehicle.nissan.patrol_2021']
TRUCK = [
    'vehicle.carlamotors.carlacola',
    'vehicle.tesla.cybertruck']
LARGE_VEHICLE = SUV + TRUCK


def _traffic_flow_scenario(wp, filters='+common', idp=1, forward_num=6, backward_num=4, **kwargs):   #idp=0.5
    # Desc: 在当前waypoint的左侧车道或者右侧车道生成车流
    results = []

    # Desc: 先向前生成车流
    _vehicle_wp = wp
    right_forward_index = 1
    while right_forward_index <= forward_num:
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            _vehicle_wp_new = carla.Transform(location = carla.Location(x=_vehicle_wp.transform.location.x ,y=_vehicle_wp.transform.location.y ,z =_vehicle_wp.transform.location.z +  1),rotation = _vehicle_wp.transform.rotation)
            results.append((bp_name, _vehicle_wp_new))
        _vehicle_wps = _vehicle_wp.next(random.randint(3, 25))
        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        right_forward_index += 1

    # Desc: 再向后生成车流
    _vehicle_wp = wp
    right_backward_index = 1
    while right_backward_index <= backward_num:
        _vehicle_wps = _vehicle_wp.previous(8)
        _vehicle_wp_new = carla.Transform(
            location=carla.Location(x=_vehicle_wp.transform.location.x, y=_vehicle_wp.transform.location.y,
                                    z=_vehicle_wp.transform.location.z + 1), rotation=_vehicle_wp.transform.rotation)

        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp_new))
        right_backward_index += 1

    return results

def choose_bp_name(filters):
    """
    Desc: 根据车辆类型和车轮数选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # Special: 类型说明
    # car: 轿车
    # suv: SUV
    # truck: 卡车
    # van: 箱型车
    # bus: 巴士
    # moto: 摩托车
    # electric: 电瓶车
    # bicycle: 自行车
    # special: 特种车辆
    # police: 警车
    # fire: 消防车
    # wheel2: 两轮车辆
    # wheel4: 四轮车辆
    # large: 大型车辆
    # small: 小型车辆
    # common: 常见车辆：排除了特种车辆和自行车和小型车辆
    # hcy1: huchuanyang自定义的车辆集合

    # e.g. +wheel4-special
    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_bp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_bp_names.extend(TYPE_VEHICLE_DICT[filter_type])
        elif op == '-':
            candidate_bp_names = list(set(candidate_bp_names) - set(TYPE_VEHICLE_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_bp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_bp_names)


def _apply_bp_generation(client, world, bp_and_transforms, name_prefix='vehicle'):
    traffic_manager = client.get_trafficmanager()
    offset_index = 0
    for v_index, (v_bp, v_transform) in enumerate(bp_and_transforms):
        v_bp = world.get_blueprint_library().find(v_bp)
        
        # 如果是车辆，确保不使用红色
        if v_bp.has_attribute('color'):
            # 获取可用的颜色选项
            color_options = v_bp.get_attribute('color').recommended_values
            # 过滤掉包含'red'的颜色选项
            safe_colors = [c for c in color_options if 'red' not in c.lower()]
            if safe_colors:
                # 随机选择一个非红色
                v_bp.set_attribute('color', random.choice(safe_colors))
        
        try:
            right_actor = world.spawn_actor(v_bp, v_transform)
            if right_actor.type_id.startswith('vehicle'):
                right_actor.set_autopilot(enabled=True)
                traffic_manager.auto_lane_change(right_actor, False)  # 禁止自动变道
                right_actor.apply_control(carla.VehicleControl(throttle=(random.randint(3, 5) / 10)))
            else:
                offset_index += 1
            continue
        except RuntimeError as e:
            if "collision at spawn position" in str(e):
                continue
            else:
                raise

def gen_traffic_flow(client,world, ego_wp):   # input :自车的wp
    # For: 右侧交通流
    right_traffic_flow_scenario(client, world, ego_wp,
                                scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'lanes_num': 1},    # 改交通流密度
                                gen_cfg={'name_prefix': 'right'}   )
    # For: 左侧交通流
    left_traffic_flow_scenario(client,world,ego_wp,
                               scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'lanes_num': 1},
                               gen_cfg={'name_prefix': 'left'})
    # For: 对向交通流
    opposite_traffic_flow_scenario(client,world,ego_wp,
                                   scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'backward_num': 3},
                                   gen_cfg={'name_prefix': 'opposite'})
    # # For: 路边停靠车辆
    # right_parking_vehicle_scenario(world,ego_wp,
    #                                scene_cfg={'filters': '+wheel4-large', 'idp': 0.0, 'forward_num': 2},
    #                                gen_cfg={'name_prefix': 'park'})

    # if actor.type_id.startswith('vehicle'):
    #     actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
    #     if a_index == 0:
    #         self.traffic_manager.set_desired_speed(actor, 0)
    #     else:
    #         self.traffic_manager.set_desired_speed(actor, random.randint(10, 15))


def right_traffic_flow_scenario(client,world, wp, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成交通流，如果右侧为行驶车道
    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
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
        _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1


def left_traffic_flow_scenario(client,world, wp, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的左侧车道生成交通流，如果左侧为行驶车道
    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
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
        _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1


def opposite_traffic_flow_scenario(client,world, wp, scene_cfg={}, gen_cfg={}):
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
    right_traffic_flow_scenario(client,world, last_wp, scene_cfg, gen_cfg)


def right_parking_vehicle_scenario(client,world, wp, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成停车车辆
    processed_lanes = set()
    if scene_cfg.get('self_lane', False):
        if wp.lane_type == carla.LaneType.Stop or (wp.lane_type == carla.LaneType.Shoulder and wp.lane_width >= 2):
            bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
            _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
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
        _apply_bp_generation(client,world, bp_and_transforms, **gen_cfg)
        processed_lanes.add(wp.lane_id)

def ahead_obstacle_scenario(world,wp, scene_cfg={}, gen_cfg={}) :    # construction 1 or construction 2
    # 检查gen_cfg字典中是否有name_prefix键，根据其值决定调用哪个函数
    name_prefix = gen_cfg.get('gen_cfg', 'construction1')  # 默认值为construction1
    if name_prefix == 'construction1':
        return ahead_obstacle_scenario_first(world, wp, scene_cfg, gen_cfg)
    elif name_prefix == 'construction2':
        return ahead_obstacle_scenario_second(world, wp, scene_cfg, gen_cfg)
    else:
        raise ValueError("Invalid generation configuration. 'name_prefix' must be 'construction1' or 'construction2'.")


def ahead_obstacle_scenario_first(world,wp, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的前方生成施工现场
    # 从场景配置字典中获取锥筒数量和间隔距离
    num_cones = scene_cfg.get('num_cones', 5)  # 默认值为5
    cone_interval = scene_cfg.get('cone_interval', 3)  # 默认值为5米
    num_garbage = scene_cfg.get('num_garbage', 50)
    num_workers = scene_cfg.get('num_workers', 4)
    # 1.生成施工牌/水马
    barrier_spawn = gen_barrier(world, wp)
    # 2.生成纸板
    ref_spawn = gen_creasedbox(world, barrier_spawn)
    # 3.生成锥筒
    gen_cones(world, barrier_spawn, num_cones, cone_interval)
    # 4.生成垃圾
    gen_garbage(world, barrier_spawn, num_garbage)
    # 5.生成行人
    # walker_manager = WalkerManager(world, num_workers, ref_spawn)
    # walker_manager.gen_walkers(num_workers, ref_spawn)
    pedestrians, controllers = gen_Walker(world, num_workers, ref_spawn)
    
    # 可以在这里保存行人和控制器的引用，以便后续清理
    return barrier_spawn.transform.location, pedestrians, controllers

def ahead_obstacle_scenario_second(world,wp, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的前方生成施工现场
    # 从场景配置字典中获取锥筒数量和间隔距离
    num_cones = scene_cfg.get('num_cones', 5)  # 默认值为5
    cone_interval = scene_cfg.get('cone_interval', 3)  # 默认值为5米
    num_workers = scene_cfg.get('num_workers', 4)
    # 1.生成施工牌/水马
    barrier_spawn = gen_barrier(world, wp)
    # 2.生成两块钢板
    ref_spawn = gen_two_planks(world, barrier_spawn)
    # 3.生成锥筒
    gen_cones(world, barrier_spawn, num_cones, cone_interval)
    # 4.生成行人
    # walker_manager = WalkerManager(world, num_workers, ref_spawn)
    # walker_manager.gen_walkers(num_workers, ref_spawn)
    gen_Walker(world,num_workers, ref_spawn)
    return barrier_spawn.transform.location


def gen_two_planks(world,barrier_spawn):
    blueprint_library = world.get_blueprint_library()
    plank_bp = choose_obsbp_name('+plank')
    print(f"plank : {plank_bp}\n")
    # 计算障碍物的位置
    plank_spawn_first = move_waypoint_forward(barrier_spawn, random.randint(2, 3))
    plank_transform_first = plank_spawn_first.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    plank_blueprint_first = blueprint_library.find(plank_bp)
    if plank_blueprint_first is not None:
        new_yaw = plank_transform_first.rotation.yaw - 20
        # 创建一个新的Transform对象，使用新的yaw值
        plank_transform_first = carla.Transform(
            location=carla.Location(x=plank_transform_first.location.x + 0.5, y=plank_transform_first.location.y - 0.5,
                                    z=plank_transform_first.location.z + 0.5),
            rotation=carla.Rotation(pitch=plank_transform_first.rotation.pitch, yaw=new_yaw,
                                    roll=plank_transform_first.rotation.roll))
        plank_first = world.spawn_actor(plank_blueprint_first, plank_transform_first)
        plank_first.set_simulate_physics(False)  # Ensure the barrier has physics simulation

    plank_spawn_second = move_waypoint_forward(barrier_spawn, random.randint(5, 7))
    plank_transform_second = plank_spawn_second.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    plank_blueprint_second = blueprint_library.find(plank_bp)
    if plank_blueprint_second is not None:
        new_yaw_second = plank_transform_second.rotation.yaw - 70
        # 创建一个新的Transform对象，使用新的yaw值
        plank_transform_second = carla.Transform(
            location=carla.Location(x=plank_transform_second.location.x - 0.3,
                                    y=plank_transform_second.location.y + 0.3,
                                    z=plank_transform_second.location.z + 0.5),
            rotation=carla.Rotation(pitch=plank_transform_second.rotation.pitch, yaw=new_yaw_second,
                                    roll=plank_transform_second.rotation.roll))
        plank_second = world.spawn_actor(plank_blueprint_second, plank_transform_second)
        plank_second.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return plank_spawn_first


def gen_barrier(world,wp):
    blueprint_library = world.get_blueprint_library()
    # 根据过滤条件选择障碍物蓝图
    barrier_bp = choose_obsbp_name('+street_barrier')
    print(f"barrier_bp : {barrier_bp}\n")
    # 计算障碍物的位置
    barrier_spawn = move_waypoint_forward(wp, random.randint(25, 30))       # debug出來 兩個wp 一樣
    barrier_transform = barrier_spawn.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    barrier_blueprint = blueprint_library.find(barrier_bp)
    if barrier_blueprint is not None:
        new_yaw = barrier_transform.rotation.yaw + 90
        # 创建一个新的Transform对象，使用新的yaw值
        new_transform = carla.Transform(
            location=barrier_transform.location,  # 保持位置不变
            rotation=carla.Rotation(pitch=barrier_transform.rotation.pitch, yaw=new_yaw,
                                    roll=barrier_transform.rotation.roll))
        barrier = world.spawn_actor(barrier_blueprint, new_transform)
        barrier.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return barrier_spawn


def choose_obsbp_name(filters):
    """
    Desc: 根据障碍物类型选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # garbage: 道路垃圾，废弃物
    # bus_stop: 公交车站
    # construction： 施工
    # street_barrier: 道路指引
    # traffic_barrier: 交通障碍物

    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_obsbp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_obsbp_names.extend(TYPE_OBSTACLE_DICT[filter_type])
        elif op == '-':
            candidate_obsbp_names = list(set(candidate_obsbp_names) - set(TYPE_OBSTACLE_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_obsbp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_obsbp_names)



def gen_creasedbox(world,barrier_spawn):
    blueprint_library = world.get_blueprint_library()
    creasedbox_bp = choose_obsbp_name('+creasedbox')
    print(f"creasedbox : {creasedbox_bp}\n")
    # 计算障碍物的位置
    creasedbox_spawn = move_waypoint_forward(barrier_spawn, random.randint(30, 35))
    creasedbox_transform = creasedbox_spawn.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    creasedbox_blueprint = blueprint_library.find(creasedbox_bp)
    if creasedbox_blueprint is not None:
        new_yaw = creasedbox_transform.rotation.yaw + 10  # 45
        # 创建一个新的Transform对象，使用新的yaw值
        new_creasedbox_transform = carla.Transform(
            location=carla.Location(x=creasedbox_transform.location.x, y=creasedbox_transform.location.y,
                                    z=creasedbox_transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=creasedbox_transform.rotation.pitch, yaw=new_yaw,
                                    roll=creasedbox_transform.rotation.roll))
        creasedbox = world.spawn_actor(creasedbox_blueprint, new_creasedbox_transform)
        creasedbox.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return creasedbox_spawn

def gen_garbage(world,barrier_spawn,num_garbage):
    blueprint_library = world.get_blueprint_library()
    gar_spawn = move_waypoint_forward(barrier_spawn, random.randint(2, 3))
    for i in range(num_garbage):
        garbage_bp = choose_obsbp_name('+garbage')
        garbage_blueprint = blueprint_library.find(garbage_bp)
        k_gar_soffset = random.choice([2, 3])
        x_gar_soffset = random.uniform(0, 0.3)
        y_gar_soffset = random.uniform(0, 0.5)
        yaw_gar_soffset = random.uniform(0, 360)
        spawn_garbage_point = carla.Transform(
            location=carla.Location(
                x=gar_spawn.transform.location.x + 1.0 + (-1) ** (-1 * k_gar_soffset) * x_gar_soffset * 0.4,
                y=gar_spawn.transform.location.y + (-1) ** (k_gar_soffset) * y_gar_soffset * 0.3,
                z=gar_spawn.transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=gar_spawn.transform.rotation.pitch,
                                    yaw=gar_spawn.transform.rotation.yaw + (-1) ** (k_gar_soffset) * yaw_gar_soffset,
                                    roll=gar_spawn.transform.rotation.roll)
        )
        while True:
            garbage = world.spawn_actor(garbage_blueprint, spawn_garbage_point)
            garbage.set_simulate_physics(False)
            break
    return gar_spawn


def gen_cones(world,barrier_spawn,num_cones,cone_interval):
    blueprint_library = world.get_blueprint_library()
    cone_bp = choose_obsbp_name('+traffic_barrier')
    cone_blueprint = blueprint_library.find(cone_bp)
    if cone_bp is None:
        raise ValueError("Traffic cone blueprint not found in the library.")

    # Get the waypoint just ahead of the barrier
    _map = world.get_map()
    barrier_waypoint = _map.get_waypoint(barrier_spawn.transform.location)
    first_cone_waypoint = barrier_waypoint.next(0.3)[0]  # Get the next waypoint after the barrier

    # Spawn the traffic cones
    all_cone_locations = []
    
    for i in range(num_cones):
        try:
            target_waypoint = first_cone_waypoint.next((i + 1) * int(cone_interval))[0]
            if target_waypoint is not None:
                assert isinstance(target_waypoint, carla.Waypoint)
                
                # 左侧锥桶
                cone_left_location = carla.Location(
                    x=target_waypoint.transform.location.x + (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    y=target_waypoint.transform.location.y - (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    z=target_waypoint.transform.location.z)
                
                # 存储左侧锥桶坐标
                all_cone_locations.append({
                    'cone': f'左侧锥桶 #{i+1}',
                    'x': round(cone_left_location.x, 2),
                    'y': round(cone_left_location.y, 2),
                    'z': round(cone_left_location.z, 2)
                })
                
                # 创建锥筒的变换对象
                cone_left_transform = carla.Transform(
                    location=cone_left_location,
                    rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                            yaw=target_waypoint.transform.rotation.yaw,
                                            roll=target_waypoint.transform.rotation.roll))
                # 在计算出的位置和方向上生成锥筒
                cone = world.spawn_actor(cone_blueprint, cone_left_transform)
                cone.set_simulate_physics(False)

                cone_right_location = carla.Location(
                    x=target_waypoint.transform.location.x - (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    y=target_waypoint.transform.location.y + (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    z=target_waypoint.transform.location.z)
                
                # 存储右侧锥桶坐标
                all_cone_locations.append({
                    'cone': f'右侧锥桶 #{i+1}', 
                    'x': round(cone_right_location.x, 2),
                    'y': round(cone_right_location.y, 2),
                    'z': round(cone_right_location.z, 2)
                })
                
                # 创建锥筒的变换对象
                cone_right_transform = carla.Transform(
                    location=cone_right_location,
                    rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                            yaw=target_waypoint.transform.rotation.yaw,
                                            roll=target_waypoint.transform.rotation.roll))
                # 在计算出的位置和方向上生成锥筒
                cone = world.spawn_actor(cone_blueprint, cone_right_transform)
                cone.set_simulate_physics(False)

            else:
                print(f"Invalid waypoint for cone placement at i = {i}")
        except RuntimeError as e:
            print(f"Error placing cones at i = {i}: {e}")

    # 在终端打印所有锥桶坐标
    print("\n所有锥桶的坐标位置:")
    print("-" * 50)
    for cone in all_cone_locations:
        print(f"{cone['cone']}:")
        print(f"    x: {cone['x']}, y: {cone['y']}, z: {cone['z']}")
    print("-" * 50)

def gen_Walker(world, num_workers, ref_spawn):
    blueprint_library = world.get_blueprint_library()
    pedestrians = []
    controllers = []  # 添加控制器列表
    pedestrians_bp = choose_walker_name('+workers')
    pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
    max_attempts = 2
    attempts = 0
    
    # 获取行人AI控制器的蓝图
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    
    for i in range(num_workers):
        spawn_point = move_waypoint_forward(ref_spawn, random.randint(0, 5))
        random_yaw = random.uniform(0, 180)
        spawn_npc_point = carla.Transform(
            location=carla.Location(x=spawn_point.transform.location.x, 
                                  y=spawn_point.transform.location.y,
                                  z=spawn_point.transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                  yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                  roll=spawn_point.transform.rotation.roll))
                                  
        while attempts < max_attempts:
            try:
                # 生成行人
                walker = world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
                # 生成并设置行人控制器
                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                
                # 设置行人行为
                controller.start()
                controller.set_max_speed(random.uniform(1.0, 2.0))  # 设置随机行走速度
                
                # 获取随机目标位置 (在当前位置附近)
                target_x = spawn_npc_point.location.x + random.uniform(-20, 20)
                target_y = spawn_npc_point.location.y + random.uniform(-20, 20)
                target_location = carla.Location(x=target_x, y=target_y, z=spawn_npc_point.location.z)
                
                # 设置行人目标位置
                controller.go_to_location(target_location)
                
                pedestrians.append(walker)
                controllers.append(controller)
                break
                
            except RuntimeError as e:
                attempts += 1
                if attempts >= max_attempts:
                    break
                spawn_point = move_waypoint_forward(ref_spawn, random.randint(3, 6))
                random_yaw = random.uniform(0, 180)
                spawn_npc_point = carla.Transform(
                    location=carla.Location(x=spawn_point.transform.location.x, 
                                          y=spawn_point.transform.location.y,
                                          z=spawn_point.transform.location.z + 0.5),
                    rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                          yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                          roll=spawn_point.transform.rotation.roll))

    return pedestrians, controllers  # 返回行人和控制器列表

def move_waypoint_forward(wp, distance):     #  有問題
    # Desc: 将waypoint沿着前进方向移动一定距离
    dist = 0
    next_wp = wp
    while dist < distance:
        next_wps = next_wp.next(1)
        if not next_wps: # or next_wps[0].is_junction:
            break
        next_wp = next_wps[0]
        dist += 1
    return next_wp


def move_waypoint_backward(wp, distance):
    # Desc: 将waypoint沿着反方向移动一定距离
    dist = 0
    next_wp = wp
    while dist < distance:
        next_wps = next_wp.previous(1)
        if not next_wps: #or next_wps[0].is_junction:
            break
        next_wp = next_wps[0]
        dist += 1
    return next_wp


def get_different_lane_spawn_transforms(world: carla.World, ego_point, spawn_points_num, radius=50, allow_same_side=True, allow_behind=False) -> List[Tuple[str, carla.Transform]]:
    """
    Desc: 获取指定半径内的所有其他车辆允许的生成位置，去除在路口里的生成点，且分车道判断
    :param world: carla.World
    :param ego: carla.Vehicle
    :param spawn_points_num: int
    :param radius: float
    :param allow_same_side: 允许在同一侧的其他车道生成
    :param allow_behind: 允许在自车后方生成
    """
    wmap = world.get_map()
    spawn_points = wmap.get_spawn_points()
    ego = ego_point.transform
    # ego_loc = ego.get_location()
    ego_loc = ego.location
   # ego_wp = wmap.get_waypoint(ego_loc)
    ego_road_id = ego_point.road_id
    ego_lane_id =ego_point.lane_id

    results = []
    for spawn_point in spawn_points:
        spawn_loc = spawn_point.location
        spawn_wp = wmap.get_waypoint(spawn_loc)
        if spawn_wp.is_junction or spawn_wp.is_intersection:
            continue
        if len(results) >= spawn_points_num:
            break
        if spawn_point.location.distance(ego_loc) > radius:
            continue

        add_sign = False
        if spawn_wp.road_id != ego_road_id:
            add_sign = True
        else:
            if spawn_wp.lane_id * ego_lane_id < 0:
                add_sign = True
            else:
                if spawn_wp.lane_id != ego_lane_id:
                    if allow_same_side:
                        add_sign = True
                else:
                    if allow_behind:
                        v1 = ego_point.transform.get_forward_vector()
                        v2 = spawn_loc - ego_loc
                        if calc_cos_between_vector(v1, v2) < 0:
                            add_sign = True

        if add_sign:
            results.append(('*vehicle*', spawn_point))

    return results


def normalize(v):
    # Desc: 将向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calc_cos_between_vector(carla_v1, carla_v2):
    # Desc: 计算两个carla.Vector3D之间的夹角cos
    v1 = np.array([carla_v1.x, carla_v1.y])
    v2 = np.array([carla_v2.x, carla_v2.y])
    v1 = normalize(v1)
    v2 = normalize(v2)
    cos_value = np.dot(v1, v2)
    return cos_value

def choose_walker_name(filters):
    """
    Desc: 根据障碍物类型选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_WALKERbp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_WALKERbp_names.extend(TYPE_WALKER_DICT[filter_type])
        elif op == '-':
            candidate_WALKERbp_names = list(set(candidate_WALKERbp_names) - set(TYPE_WALKER_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_WALKERbp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_WALKERbp_names)

def get_sidewalk_wps(ori_wp):
    wp = ori_wp.get_right_lane()
    while wp is not None and wp.lane_type != carla.LaneType.Sidewalk:
        wp = wp.get_right_lane()
    if wp is None:
        right_sidewalk_wp = None
    else:
        right_sidewalk_wp = wp

    wp = ori_wp.get_left_lane()
    direction_change = False
    while wp is not None and wp.lane_type != carla.LaneType.Sidewalk:
        if wp.lane_id * ori_wp.lane_id < 0:
            direction_change = True
        if direction_change:
            wp = wp.get_right_lane()
        else:
            wp = wp.get_left_lane()
    if wp is None:
        left_sidewalk_wp = None
    else:
        left_sidewalk_wp = wp
    return left_sidewalk_wp, right_sidewalk_wp

def gen_ai_walker(world, transform, CarlaDataProvider):
    pedestrian = CarlaDataProvider.request_new_actor(world, 'walker.*', transform)
    if pedestrian is not None:
        controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        controller = world.spawn_actor(controller_bp, pedestrian.get_transform(), pedestrian)
        return pedestrian, controller
    else:
        return None, None



def get_waypoint_in_distance(waypoint, distance):
    """
    Obtain a waypoint in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.
    @return obtained waypoint and the traveled distance
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance


def get_sidewalk_transform(waypoint, offset):
    """
    Processes the waypoint transform to find a suitable spawning one at the sidewalk.
    It first rotates the transform so that it is pointing towards the road and then moves a
    bit to the side waypoint that aren't part of sidewalks, as they might be invading the road
    """

    new_rotation = waypoint.transform.rotation
    new_rotation.yaw += offset['yaw']

    if waypoint.lane_type == carla.LaneType.Sidewalk:
        new_location = waypoint.transform.location
    else:
        right_vector = waypoint.transform.get_right_vector()
        offset_dist = waypoint.lane_width * offset["k"]
        offset_location = carla.Location(offset_dist * right_vector.x, offset_dist * right_vector.y)
        new_location = waypoint.transform.location + offset_location
    new_location.z += offset['z']

    return carla.Transform(new_location, new_rotation)

def get_random_pedestrian_transforms(world: carla.World, ego, spawn_points_num, debug=False) -> List[Tuple[str, carla.Transform, carla.Location]]:
    ego_wp = world.get_map().get_waypoint(ego.get_location())
    right_ref_wps = ego_wp.previous_until_lane_start(1)
    if len(right_ref_wps) > 2:
        _, right_sidewalk_wp = get_sidewalk_wps(right_ref_wps[-2])
    else:
        _, right_sidewalk_wp = get_sidewalk_wps(ego_wp)
    left_ref_wps = ego_wp.next_until_lane_end(1)
    if len(left_ref_wps) > 2:
        left_sidewalk_wp, _ = get_sidewalk_wps(left_ref_wps[-2])
    else:
        left_sidewalk_wp, _ = get_sidewalk_wps(ego_wp)
    candidate_wps = []
    if left_sidewalk_wp is not None:
        candidate_wps.extend(left_sidewalk_wp.next_until_lane_end(1))
    if right_sidewalk_wp is not None:
        candidate_wps.extend(right_sidewalk_wp.next_until_lane_end(1))
    random.shuffle(candidate_wps)

    if debug:
        debug = world.debug
        for wp in ego_wp.previous_until_lane_start(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 0, 255))
        for wp in ego_wp.next_until_lane_end(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 255, 255))
        debug.draw_point(right_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        debug.draw_point(left_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        for wp in candidate_wps:
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(255, 0, 0))
    if len(candidate_wps) > 0:
        if len(candidate_wps) > spawn_points_num:
            candidate_wps = random.sample(candidate_wps, spawn_points_num)
        else:
            candidate_wps = random.sample(candidate_wps, len(candidate_wps))
    else:
        return []

    return [('*walker*', wp.transform, choice(candidate_wps).transform.location) for wp in candidate_wps]

def get_random_pedestrian_transforms(world: carla.World, ego_point, spawn_points_num, debug=False) -> List[Tuple[str, carla.Transform, carla.Location]]:
#    ego_wp = world.get_map().get_waypoint(ego.get_location())
    ego_wp = ego_point
    right_ref_wps = ego_wp.previous_until_lane_start(1)
    if len(right_ref_wps) > 2:
        _, right_sidewalk_wp = get_sidewalk_wps(right_ref_wps[-2])
    else:
        _, right_sidewalk_wp = get_sidewalk_wps(ego_wp)
    left_ref_wps = ego_wp.next_until_lane_end(1)
    if len(left_ref_wps) > 2:
        left_sidewalk_wp, _ = get_sidewalk_wps(left_ref_wps[-2])
    else:
        left_sidewalk_wp, _ = get_sidewalk_wps(ego_wp)
    candidate_wps = []
    if left_sidewalk_wp is not None:
        candidate_wps.extend(left_sidewalk_wp.next_until_lane_end(1))
    if right_sidewalk_wp is not None:
        candidate_wps.extend(right_sidewalk_wp.next_until_lane_end(1))
    random.shuffle(candidate_wps)

    if debug:
        debug = world.debug
        for wp in ego_wp.previous_until_lane_start(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 0, 255))
        for wp in ego_wp.next_until_lane_end(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 255, 255))
        debug.draw_point(right_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        debug.draw_point(left_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        for wp in candidate_wps:
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(255, 0, 0))
    if len(candidate_wps) > 0:
        if len(candidate_wps) > spawn_points_num:
            candidate_wps = random.sample(candidate_wps, spawn_points_num)
        else:
            candidate_wps = random.sample(candidate_wps, len(candidate_wps))
    else:
        return []

    return [('*walker*', wp.transform, choice(candidate_wps).transform.location) for wp in candidate_wps]



def vehicle_breakdown(client, world,ego_location):    # 補充這一輛車的速度 0
    """
       Simulates a vehicle breakdown scenario by spawning a stationary vehicle and a barrier in front of the ego vehicle.

       :param world: The CARLA world object.
       :param ego_location: The location of the ego vehicle.
       """

    # For: 在自车前方20-25米随机生成一辆车
    first_vehicle_wp = move_waypoint_forward(ego_location, random.randint(5, 8))
    front_actor = CarlaDataProvider.request_new_actor(world, '+wheel4', first_vehicle_wp.transform)
    # traffic_manager = CarlaDataProvider.get_trafficmanager()
    traffic_manager = client.get_trafficmanager()
    # traffic_manager.set_desired_speed(front_actor, 0)
    front_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

    blueprint_library = world.get_blueprint_library()
    # 根据过滤条件选择障碍物蓝图
    barrier_bp = choose_obsbp_name('+traffic_barrier')
    print(f"barrier_bp : {barrier_bp}\n")
    # 计算障碍物的位置
    barrier_transform = move_waypoint_backward(first_vehicle_wp, random.randint(2, 3))
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    barrier_blueprint = blueprint_library.find(barrier_bp)
    if barrier_blueprint is not None:
        barrier = world.spawn_actor(barrier_blueprint, barrier_transform.transform)
        barrier.set_simulate_physics(True)  # Ensure the barrier has physics simulation
        return barrier_transform.transform.location
    else:
        print("Warning: Barrier blueprint not found.")

def front_brake(world, ego_wp):
    _first_vehicle_wp = move_waypoint_forward(ego_wp, random.randint(15, 20))
    front_bp_name = choose_bp_name('+wheel4')
    
    # 获取车辆蓝图
    front_bp = world.get_blueprint_library().find(front_bp_name)
    
    # 强制设置车辆颜色为黑色
    if front_bp.has_attribute('color'):
        front_bp.set_attribute('color', '0,0,0')
    
    # 使用修改后的蓝图生成车辆
    front_actor = world.spawn_actor(front_bp, _first_vehicle_wp.transform)
    
    # 停下来的动作  前车开的很慢很慢
    front_actor.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))


def junction_pedestrian(world, ego_wp):
  #  traffic_manager = CarlaDataProvider.get_trafficmanager()
    nearby_spawn_points = get_different_lane_spawn_transforms(world, ego_wp, random.randint(5, 30),
                                                              60, allow_same_side=True, allow_behind=True)
    pedestrian_end_wp1, pedestrian_end_wp2 = get_sidewalk_wps(ego_wp)
    if pedestrian_end_wp1 is None:
        pedestrian_end_wp = pedestrian_end_wp2
    else:
        pedestrian_end_wp = pedestrian_end_wp1
    pedestrian_start_wp = ego_wp.next_until_lane_end(1)[-1]
    _, pedestrian_start_wp = get_sidewalk_wps(pedestrian_start_wp)

    for i in range(random.randint(10, 15)):
        pedestrian, controller = gen_ai_walker(world, pedestrian_start_wp.transform, CarlaDataProvider)
        controller.start()
        #controller.go_to_location(pedestrian_end_wp.transform.location)   # 行人走路没有设置目的地
        controller.set_max_speed(random.randint(10, 30) / 10.0)
        pedestrian_start_wp = pedestrian_start_wp.previous(1.0)[0]

    for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
        npc_actor = CarlaDataProvider.request_new_actor(world, v_bp, v_transform)


def gen_emergency(world, ego_wp):   # 這個場景要注意下 LLM or reward 要單讀給一點
    emergency_behind_distance = 45
    emergency_vehicle_names = ['vehicle.dodge_charger.police']
    _first_vehicle_wp = move_waypoint_backward(ego_wp, emergency_behind_distance)
    vehicle_model_name = choice(emergency_vehicle_names)
    emergency_actor = CarlaDataProvider.request_new_actor(vehicle_model_name, _first_vehicle_wp.transform)

def gen_ghosta(world, ego_wp):
    nearby_ped_spawn_points = get_random_pedestrian_transforms(world, ego_wp, random.randint(5, 10))
    waypoint = ego_wp
    wp_next = waypoint.get_right_lane()
    pedestrian_distance = 13.75
    _num_lane_changes = 0
    _adversary_type = 'walker.*'
    other_actors = []
    actor_desc = []
    pedestrians = []
    controllers = []

    wp_pedestrian, _ = get_waypoint_in_distance(wp_next, pedestrian_distance)
    
    sidewalk_waypoint = wp_pedestrian
    while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
        right_wp = sidewalk_waypoint.get_right_lane()
        if right_wp is None:
            break
        sidewalk_waypoint = right_wp
        _num_lane_changes += 1

    offset = {'yaw': 270, 'z': 0.5, 'k': 1.0}
    _adversary_transform = get_sidewalk_transform(sidewalk_waypoint, offset)

    # 生成主要的对抗行人
    adversary = CarlaDataProvider.request_new_actor(world, _adversary_type, _adversary_transform)
    adversary.set_simulate_physics(enabled=True)
    other_actors.append(adversary)
    actor_desc.append('ghost_pedestrian')

    # 生成其他行人
    for p_index, (p_bp, p_transform, p_dest_loc) in enumerate(nearby_ped_spawn_points):
        pedestrian, controller = gen_ai_walker(world, p_transform, CarlaDataProvider)
        if pedestrian is not None:
            controller.start()
            controller.set_max_speed(0)  # 初始速度设为0
            pedestrians.append(pedestrian)
            controllers.append(controller)
            
    def check_distance_and_trigger(pedestrians_list, controllers_list):
        try:
            ego_vehicles = world.get_actors().filter('vehicle.tesla.model3')
            if not ego_vehicles:
                return
            ego_vehicle = ego_vehicles[0]
            
            for ped, cont in zip(pedestrians_list, controllers_list):
                if not ped.is_alive or not cont.is_alive:
                    continue
                    
                # 计算与自车的距离
                distance = ped.get_location().distance(ego_vehicle.get_location())
                
                if distance < 10.0:  # 当距离小于10米时触发横穿
                    # 获取对面人行道的位置
                    opposite_sidewalk = get_opposite_sidewalk_location(ped.get_location(), world)
                    if opposite_sidewalk:
                        cont.set_max_speed(random.randint(10, 30) / 10.0)  # 设置行走速度
                        cont.go_to_location(opposite_sidewalk)  # 设置目标位置
        except Exception as e:
            print(f"Error in check_distance_and_trigger: {e}")

    # 设置定时器来检查距离
    world.on_tick(lambda w: check_distance_and_trigger(pedestrians, controllers))

def obstacle_ahead(world, ego_wp):
    obstacle_location = ahead_obstacle_scenario(world,ego_wp,
                                    scene_cfg={ 'num_cones': 10,  # 想要生成的锥筒数量
                                                'cone_interval': 2.5 , # 锥筒之间的间隔距离（米）
                                                'num_garbage': 10 ,   # 想要生成的垃圾数量
                                                'num_workers': 4 },   # 想要生成的行人数量
                                    gen_cfg={'name_prefix': 'construction1'})     # 默认construction 1    construction 1 or construction 2


def preceding_vehicle(client, world, ego_wp):
    vehicle_breakdown(client, world, ego_wp)   #ego_Location

def frontbrake(world, ego_wp):
    front_brake(world, ego_wp) # 待修改前車的動作

def junctionpedestriancross(world, ego_wp):
    junction_pedestrian(world, ego_wp)


def giveaway_emergency(world, ego_wp):
    gen_emergency(world, ego_wp)


def ghostA(world, ego_wp):
    gen_ghosta(world, ego_wp)
