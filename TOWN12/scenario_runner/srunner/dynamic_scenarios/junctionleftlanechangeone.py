from __future__ import print_function


import numpy as np
import py_trees
import carla
import time

import re
from random import choice
from agents.navigation.local_planner import RoadOption
import time

from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy, SwitchWrongDirectionTest, BasicAgentBehavior, ScenarioTimeout, Idle, WaitForever, HandBrakeVehicle, OppositeActorFlow)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, WaitEndIntersection)
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior
from TOWN12.scenario_runner.srunner.dynamic_scenarios.functions.record_data import RecordData

from TOWN12.town12_tools.explainable_utils import *
from .functions import *
from .basic_desc import *
from beta.tool_wenyang.tool_meta.meta_description import _generate_scenario_desc_balance
# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class JunctionLeftLaneChangeOne(BasicScenario):
    """
    Desc: 接近路口向左变道一个车道，为左转做准备
    Special: 补充左变道的数据
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180, uniad=False, interfuser=False):
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self.timeout = timeout

        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.predefined_vehicle_length = 8
        self.actor_desc = []
        self.traffic_manager = None

        self.min_lane_change_distance = 25

        # For: tick autopilot
        self.ego_moved = False
        self.left_car = None
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 20
        self.navigation_cmds = ['Left']

        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], KEEP_S, [], ''),
            '#fix3': (LEFT_C, [], DEC, [], ''),
            '#dynamic1': (KEEP_L, [], STOP, [], ''),
            '#dynamic2': (KEEP_L, [], STOP, [], ''),
            '#fix4': (KEEP_L, [], STOP, [], ''),
            '#fix5': (KEEP_L, [], STOP, [], ''),
            '#fix6': (KEEP_L, [], DEC, [], ''),
            '#fix7': (LEFT_C, [], DEC, [], ''),
            '#fix8': (LEFT_C, [], STOP, [], ''),
        }

        super().__init__("JunctionLeftLaneChangeOne", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'

        if len(set(self.passed_lane_ids)) == 1:
            return False, 'Autopilot未变道'

        return True, ''

    def uniad_tick_autopilot(self, max_speed):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.left_car is None and 'left' in self.actor_desc:
            self.left_car = self.other_actors[self.actor_desc.index('left')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(max_speed)
            self._set_ego_autopilot()
            self.initialized = True

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        # For: 有时候Autopilot不会变道，这里强制变道以提高成功率
        if not self.lane_changed:
            if distance_to_next_junction(cur_wp) < self.min_lane_change_distance:
                self._force_ego_lanechange_left()
                self.lane_changed = True
                print(f'\nForce change lane to left')

        # Desc: 已经到达路口
        if cur_wp.is_junction or cur_wp.is_intersection:
            self._tf_set_ego_speed(max_speed)
        # Desc: 还没到达路口
        else:
            # Desc: 还在当前车道
            if cur_wp.lane_id == self.starting_wp.lane_id:
                # Desc: 还没开始变道
                if not self.lane_changed:
                    self._tf_set_ego_speed(max_speed)
                # Desc: 已经开始变道
                else:
                    if ego_speed < 0.1:
                        self._tf_set_ego_speed(20)
                    else:
                        self._tf_set_ego_speed(20)
                        self.lane_changed = True
            else:
                # Desc: 已经完成变道
                if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self._tf_set_ego_speed(max_speed)
                else:
                    self._tf_set_ego_speed(20)

        return {'cur_wp': cur_wp, 'navi_cmd': 'turn left'}

    def _get_navi_route(self, cur_wp):
        if cur_wp.is_junction or cur_wp.is_intersection:
            start_wp = cur_wp
        else:
            if cur_wp.lane_id == self.starting_wp.lane_id:
                start_wp = cur_wp.get_left_lane()
            else:
                start_wp = cur_wp
        dest_wp_start_wp = self.starting_wp.get_left_lane()
        dest_wp = get_junction_turn_after_wp(get_next_junction(dest_wp_start_wp), dest_wp_start_wp, 'left', None).next(5)[0].next(70)[0]
        return start_wp, dest_wp

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.left_car is None and 'left' in self.actor_desc:
            self.left_car = self.other_actors[self.actor_desc.index('left')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._set_ego_autopilot()
            self.initialized = True

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors),
            'actors_desc': self.actor_desc,
        }

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        RecordData.record_data(ego, cur_wp, self._world, self.other_actors)

        global last_run_time
        global last_output
        #
        current_time = time.time()
        # # 检查是否已经过了60秒
        if current_time - last_run_time >= 12:
            # 更新上次运行的时间
            last_run_time = current_time
            print("LLM starts!!!")
            RecordData._monitor_loop()
            time.sleep(2)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        # For: 有时候Autopilot不会变道，这里强制变道以提高成功率
        if not self.lane_changed:
            if distance_to_next_junction(cur_wp) < self.min_lane_change_distance:
                self._force_ego_lanechange_left()
                self.lane_changed = True
                print(f'\nForce change lane to left')

        info = f'Speed: {int(ego_speed)} km/h '

        # Desc: 已经到达路口
        if cur_wp.is_junction or cur_wp.is_intersection:
            ego_stage = '#fix1'
            ego_reason = ''
            ego_action = 'keep|junction'
            explainable_desc = f'正常行驶，通过路口'
            self._tf_set_ego_speed(20)
        # Desc: 还没到达路口
        else:
            # Desc: 还在当前车道
            if cur_wp.lane_id == self.starting_wp.lane_id:
                # Desc: 还没开始变道
                if not self.lane_changed:
                    ego_stage = '#fix2'
                    ego_reason = ''
                    ego_action = 'keep'
                    explainable_desc = f'正常行驶'
                    self._tf_set_ego_speed(20)
                # Desc: 已经开始变道
                else:
                    if ego_speed < 0.1:
                        ego_stage = '#fix8'
                        self._tf_set_ego_speed(15)
                    else:
                        ego_stage = '#fix3'
                        ego_reason = '前方需要左转'
                        ego_action = 'decelerate|lanechange|left'
                        explainable_desc = f'由于{ego_reason}，向左变道'
                        self._tf_set_ego_speed(15)
                        self.lane_changed = True
            else:
                # Desc: 已经完成变道
                if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    if ego_speed < 0.1 and self.left_car is not None and math.sqrt(self.left_car.get_velocity().x ** 2 + self.left_car.get_velocity().y ** 2) * 3.6 < 0.1 and cur_ego_loc.distance(self.left_car.get_location()) < 10:
                        desc = Edict(explainable_data['actors'][self.left_car.id]['description'])
                        if junction_has_traffic_light(self._world, cur_wp):
                            ego_stage = '#dynamic1'
                            ego_reason = '等待红绿灯'
                            ego_action = 'stop|wait'
                            explainable_desc = f'{desc.direction}{desc.distance}处有一辆{desc.color}{desc.type}{ego_reason}，停车等待'
                        else:
                            ego_stage = '#dynamic2'
                            ego_reason = '前车静止'
                            ego_action = 'stop|wait'
                            explainable_desc = f'{desc.direction}{desc.distance}处有一辆{desc.color}{desc.type}{ego_reason}静止，停车等待'
                    elif ego_speed < 0.1 and self.left_car is None:
                        if junction_has_traffic_light(self._world, cur_wp):
                            ego_stage = '#fix4'
                            ego_reason = '等待红绿灯'
                            ego_action = 'stop|wait'
                            explainable_desc = f'停车等待红绿灯'
                        else:
                            ego_stage = '#fix5'
                            ego_reason = '路口停止标志'
                            ego_action = 'stop|wait'
                            explainable_desc = f'由于{ego_reason}，观察四周'
                    else:
                        ego_stage = '#fix6'
                        ego_reason = '前方左转|接近路口'
                        ego_action = 'decelerate'
                        explainable_desc = f'由于{ego_reason}，减速'
                        self._tf_set_ego_speed(10)
                else:
                    ego_stage = '#fix7'
                    ego_reason = '前方需要左转'
                    ego_action = 'decelerate|lanechange|left'
                    explainable_desc = f'由于{ego_reason}，向左变道'
                    self._tf_set_ego_speed(15)

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = 'turn left'
        explainable_data['info'] = info

        meta_input = {'sce_name': self.name, 'chuyang_inte': explainable_data}
        reason = _generate_scenario_desc_balance(meta_input)['description_demo_eng']

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{reason}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        print(f'\r{info}', end='')

        return explainable_data

    def _initialize_actors(self, config):
        # Depend: 场景保证左侧存在可行驶车道
        lane_info = get_lane_info(self.starting_wp)

        # For: 80%概率在自车前方生成一辆车，同时要满足距离路口大于15米
        if self.distance_to_junction > 15 and random.random() < 0.8:
            _first_vehicle_wp = move_waypoint_forward(self.starting_wp, random.randint(self.predefined_vehicle_length, int(self.distance_to_junction - self.predefined_vehicle_length // 2)))
            front_actor = CarlaDataProvider.request_new_actor('*vehicle*', _first_vehicle_wp.transform)
            self.other_actors.append(front_actor)
            self.actor_desc.append('front')

        # For: 50%概率在左侧车道前方生成一辆车，同时要满足距离路口大于15米
        left_lane = self.starting_wp.get_left_lane()
        if self.distance_to_junction > 15 and random.random() < 0.9:
            left_vehicle_wp = move_waypoint_forward(left_lane, random.randint(self.predefined_vehicle_length, int(self.distance_to_junction - self.predefined_vehicle_length // 2)))
            left_actor = CarlaDataProvider.request_new_actor('*vehicle*', left_vehicle_wp.transform)
            self.other_actors.append(left_actor)
            self.actor_desc.append('left')

        # For: 右侧如果存在车道，80%的概率生成车流
        if lane_info.r2l > 1 and random.random() < 0.9:
            right_vehicles = right_lane_vehicle_flow_scenario(self.starting_wp)
            for v_index, (v_bp, v_transform) in enumerate(right_vehicles):
                right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
                self.other_actors.append(right_actor)
                self.actor_desc.append(f'right_{v_index}')

        # For: 生成NPC
        nearby_spawn_points = get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 3))
        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        # For: 这里设置autopilot
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_desc, actor in zip(self.actor_desc, self.other_actors):
            actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
            if 'npc' in a_desc:
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))
                self.traffic_manager.auto_lane_change(actor, False)
            else:
                self.traffic_manager.set_desired_speed(actor, 15)

    def _create_behavior(self):
        """
        The vehicle has to drive the reach a specific point but an accident is in the middle of the road,
        blocking its route and forcing it to lane change.
        """
        root = py_trees.composites.Sequence(name="JunctionLeftLaneChangeOne")
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        # criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        criteria = []
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass
