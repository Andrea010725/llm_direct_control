from __future__ import print_function

import random

# import ipdb
import numpy as np
import py_trees
import carla
import re
from random import choice
import operator
from agents.navigation.local_planner import RoadOption
import time

from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy, SwitchWrongDirectionTest, BasicAgentBehavior, ScenarioTimeout, Idle, WaitForever, HandBrakeVehicle, OppositeActorFlow)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, WaitEndIntersection, InTriggerDistanceToVehicle, InTriggerDistanceToNextIntersection)
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior
from TOWN12.scenario_runner.srunner.dynamic_scenarios.functions.record_data import RecordData

from TOWN12.town12_tools.explainable_utils import *
from .functions import *
from .basic_desc import *
# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class OvertakeVehicleFromLeft(BasicScenario):
    """
    Desc: TODO
    Special: 补充TODO的数据
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

        # For: tick autopilot
        self.front_car = None
        self.front_car_wp = None
        self.front_car_loc = None
        self.initialized = False
        self.lane_changed = False
        self.stage = 0
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        self.obstacle_index = -1
        
        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#dynamic1': (LEFT_C, [], KEEP_S, [], ''),
            '#dynamic2': (LEFT_C, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], ACC, [], ''),
            '#dynamic3': (LEFT_C, [], KEEP_S, [], ''),
            '#fix3': (KEEP_L, [], ACC, [], ''),
            '#dynamic4': (RIGHT_C, [], KEEP_S, [], ''),
            '#fix4': (KEEP_L, [], ACC, [], ''),
            '#dynamic5': (RIGHT_C, [], DEC, [], ''),
            '#fix5': (KEEP_L, [], KEEP_S, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("OvertakeVehicleFromLeft", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'
        change_num = 0
        lane_id = self.passed_lane_ids[0]
        other_lane_id = -1
        for item in self.passed_lane_ids[1:]:
            if item != lane_id:
                change_num += 1
                lane_id = item
            if item != self.starting_wp.lane_id and other_lane_id == -1:
                other_lane_id = item
        if change_num != 2:
            return False, f'Autopilot变道次数不为2次，实际变道次数为{change_num}次'
        # if other_lane_id < self.starting_wp.lane_id:
            # return False, f'Autopilot未向左变道'
        return True, ''

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]
            self.front_car_wp = self._map.get_waypoint(self.front_car.get_location())
            self.front_car_loc = self.front_car.get_location()

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors, eng=True),
            'actors_desc': self.actor_desc,
        }

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)
        front_car_wp = self._map.get_waypoint(self.front_car.get_location())

        RecordData.record_data(ego, cur_wp, self._world, self.other_actors)

        global last_run_time
        global last_output
        #
        current_time = time.time()
        # # 检查是否已经过了60秒
        if current_time - last_run_time >= 10:
            # 更新上次运行的时间
            last_run_time = current_time
            print("LLM starts!!!")
            RecordData._monitor_loop()
            time.sleep(2)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        info = f'Speed: {int(ego_speed)} km/h '

        if self.front_car.is_alive:
            desc = Edict(explainable_data['actors'][self.front_car.id]['description'])
            data = Edict(explainable_data['actors'][self.front_car.id])
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)
            info += f'Distance: {distance}m '

            # Special: 第一阶段：向左变道
            if self.stage == 0:
                # Desc: 在同一车道
                if cur_wp.lane_id == front_car_wp.lane_id:
                    if distance > 35:
                        ego_stage = '#fix1'
                        reason = 'because there are no special circumstances, so normal driving.'
                        self._tf_set_ego_speed(20)
                    elif 35 >= distance > 20:
                        ego_stage = '#dynamic1'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego and there are {right_car_num} vehicles on the right side, so prepare to overtake on the left side.'
                        else:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego, so prepare to overtake on the left side.'
                        self._tf_set_ego_speed(20)
                    elif distance <= 20:
                        if self.lane_changed is False:
                            # self._tf_set_ego_offset(-cur_wp.lane_width)
                            # self._tf_set_ego_force_go(100)
                            self._force_ego_lanechange_left()
                            print(f'\n>>> force lane change to left')
                            self.lane_changed = True
                            self._tf_set_ego_speed(20)

                        ego_stage = '#dynamic2'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego and there are {right_car_num} vehicles on the right side, change lanes to the left and overtake the slow-moving {desc.type}.'
                        else:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego, so change lanes to the left and overtake the slow-moving {desc.type}.'
                        self._tf_set_ego_speed(20)
                    else:
                        print(f'Error: distance = {distance}m')
                        raise NotImplementedError
                else:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        # Desc: 完成变道
                        ego_stage = '#fix2'
                        reason = f'because there is a {desc.color} {desc.type} moving at the {desc.direction} of ego, so overtaking the {desc.type}.'
                        self._tf_set_ego_speed(40)
                    else:
                        # Desc: 还未完成变道
                        ego_stage = '#dynamic3'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego and there are {right_car_num} vehicles on the right side, so change lanes to the left and overtake the {desc.speed} {desc.type}.'
                        else:
                            reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego, so change lanes to the left and overtake the {desc.speed} {desc.type}.'
                        self._tf_set_ego_speed(20)

                # Special: 判断是否完成第一阶段
                if self.lane_changed and cur_wp.lane_id != front_car_wp.lane_id and cur_ego_loc.distance(
                        cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 1
                    self.lane_changed = False

            # Special: 第二阶段：超车中
            elif self.stage == 1:
                if cur_wp.lane_id == front_car_wp.lane_id:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        # Desc: 完成超车
                        ego_stage = '#fix3'
                        reason = f'because there are no special circumstances, so normal driving.'
                        self._tf_set_ego_speed(40)
                    else:
                        # Desc: 还没完成返回原车道
                        ego_stage = '#dynamic4'
                        reason = f'because ego has overtaken the {desc.color}, {desc.speed} {desc.type} at the {desc.direction}, so change lanes to the right and about to complete the overtaking.'
                else:
                    # Desc: 还在左侧车道
                    if distance > 7 and actor_front_of_ego(data) is False and self.lane_changed is False:
                        # self._tf_set_ego_offset(0)
                        # self._tf_set_ego_force_go(0)
                        if ego_speed < 25:
                            # self._force_ego_lanechange_right()
                            self._tf_set_ego_offset(cur_wp.lane_width)
                            self.lane_changed = True
                            print(f'\n>>> force lane change to right')
                        self._tf_set_ego_speed(20)
                        info += f'减速 '
                    else:
                        self._tf_set_ego_speed(40)
                        info += f'加速 '

                    if self.lane_changed is False:
                        ego_stage = '#fix4'
                        reason = f'because there is a {desc.color}, {desc.speed} {desc.type}  at the {desc.direction} of ego, so accelerate and overtake.'
                    else:
                        ego_stage = '#dynamic5'
                        reason = f'because ego has overtaken {desc.color}, {desc.speed} {desc.type}  at the {desc.direction}, so change lanes to the right and about to complete the overtaking.'

                # Special: 判断是否完成第二阶段
                if self.lane_changed and cur_wp.lane_id == front_car_wp.lane_id and cur_ego_loc.distance(
                        cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 2
                    self.lane_changed = False

            # Special: 第三阶段：变道回原始车道
            elif self.stage == 2:
                ego_stage = '#fix5'
                reason = f'because there are no special circumstances, so normal driving.'
                self._tf_set_ego_speed(40)
            else:
                raise NotImplementedError
        else:
            self._tf_set_ego_speed(40)
            ego_stage = '#fix6'
            reason = f'because there are no special circumstances, so normal driving.'

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = 'follow lane'
        explainable_data['info'] = info

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{env_description}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        # Depend: 场景保证左侧可以超车
        lane_info = get_lane_info(self.starting_wp)

        # For: 生成NPC
        nearby_spawn_points = get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 5))

        # For: 在自车前方50米生成一辆车
        _first_vehicle_wp = move_waypoint_forward(self.starting_wp, random.randint(30, 40))
        front_actor = CarlaDataProvider.request_new_actor('*vehicle*', _first_vehicle_wp.transform)
        self.other_actors.append(front_actor)
        self.actor_desc.append('front')
        self.front_index = len(self.other_actors) - 1

        # For: 右侧如果存在车道，60%的概率生成车流
        if lane_info.r2l > 1 and random.random() < 0.6:
            right_vehicles = right_lane_vehicle_flow_scenario(self.starting_wp)
            for v_index, (v_bp, v_transform) in enumerate(right_vehicles):
                right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
                self.other_actors.append(right_actor)
                self.actor_desc.append(f'right_{v_index}')

        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, actor in enumerate(self.other_actors):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                if a_index == 0:
                    self.traffic_manager.set_desired_speed(actor, choice([5, 10]))
                else:
                    self.traffic_manager.set_desired_speed(actor, random.randint(10, 15))
    
    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="OvertakeVehicleFromLeft", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="OvertakeVehicleFromLeft_c1")
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 10))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 50, comparison_operator=operator.gt))
        root.add_child(c1)
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass

    