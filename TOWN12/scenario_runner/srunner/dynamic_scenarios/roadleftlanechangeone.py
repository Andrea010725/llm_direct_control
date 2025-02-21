from __future__ import print_function

import traceback

# import ipdb
# import ipdb
import numpy as np
import py_trees
import carla
import re
import operator
import time

from random import choice
from agents.navigation.local_planner import RoadOption

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
from beta.tool_wenyang.tool_meta.meta_description import _generate_scenario_desc_balance
# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class RoadLeftLaneChangeOne(BasicScenario):
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

        self.front_car_distance = 50

        # For: tick autopilot
        self.ego_moved = False
        self.front_car = None
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 40
        self.navigation_cmds = ['Straight']
        self.front_index = -1

        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], '前车距离我40米以上'),
            '#dynamic1': (KEEP_L, [], DEC, ['vehicle.N'], '前车距离我30-40米'),
            '#dynamic2_1': (LEFT_C, ['vehicle.N'], DEC, ['vehicle.N'], '前车距离我25-30米，准备变道'),
            '#dynamic2_2': (LEFT_C, ['vehicle.N', 'vehicle.E'], DEC, ['vehicle.N'], '前车距离我25-30米&右侧有车，准备变道'),
            '#dynamic3_1': (LEFT_C, ['vehicle.N'], KEEP_S, [], '前车距离我25米内，变道'),
            '#dynamic3_2': (LEFT_C, ['vehicle.N', 'vehicle.E'], KEEP_S, [], '前车距离我25米内&右侧有车，变道'),
            '#dynamic4_1': (LEFT_C, ['vehicle.N'], KEEP_S, [], '已经切换车道，但是没完成变道'),
            '#dynamic4_2': (LEFT_C, ['vehicle.N', 'vehicle.E'], KEEP_S, [], '已经切换车道，但是没完成变道&右侧有车'),
            '#fix2_1': (KEEP_L, [], ACC, [], '已经完成变道，加速'),
            '#fix2_2': (KEEP_L, [], KEEP_S, [], '已经完成变道，保持速度'),
            '#fix3': (KEEP_L, [], KEEP_S, [], '场景完成，保持速度')
        }

        super().__init__("RoadLeftLaneChangeOne", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'

        if len(set(self.passed_lane_ids)) == 1:
            return False, 'Autopilot未变道'

        return True, ''

    def uniad_tick_autopilot(self, max_speed):
        ego = self.ego_vehicles[0]

        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(max_speed)
            self._set_ego_autopilot()
            self.initialized = True

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)
        front_car_wp = self._map.get_waypoint(self.front_car.get_location())

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        if self.front_car.is_alive:
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)

            # Desc: 在同一车道
            if cur_wp.lane_id == front_car_wp.lane_id:
                # Desc: 前车在前方20米外
                if distance >= 20:
                    self._tf_set_ego_speed(max_speed)

                # Desc: 前车在前方25-30米
                elif distance > 15:
                    self._tf_set_ego_speed(20)

                # Desc: 前车在前方25米内
                elif distance <= 15:
                    if not self.lane_changed:
                        self.traffic_manager.force_lane_change(ego, False)
                        print(f'\n#################### force lane change to left')
                        self.lane_changed = True
                    self._tf_set_ego_speed(20)

                # Desc: 距离异常
                else:
                    print(f'Error: distance = {distance}m')
                    raise NotImplementedError
            else:
                # Desc: 还未完成变道
                if cur_ego_loc.distance(cur_wp.transform.location) > cur_wp.lane_width / 4.0:
                    self._tf_set_ego_speed(20)

                # Desc: 已经完成变道
                else:
                    self._tf_set_ego_speed(max_speed)
        else:
            self._tf_set_ego_speed(max_speed)

        return {'cur_wp': cur_wp, 'navi_cmd': 'follow lane'}

    def _get_navi_route(self, cur_wp):
        if cur_wp.lane_id == self.starting_wp.lane_id:
            start_wp = cur_wp
        else:
            start_wp = cur_wp.get_right_lane()
        dest_wp = self.starting_wp.next_until_lane_end(1)[-2].next(70)[0]
        return start_wp, dest_wp

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]

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
        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])

        if self.front_car.is_alive:
            # desc = Edict(explainable_data['actors'][self.front_car.id]['description'])
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)
            info += f'Front car distance: {distance}m '

            # Desc: 在同一车道
            if cur_wp.lane_id == front_car_wp.lane_id:
                # Desc: 前车在前方40米以上
                if distance > 40:
                    ego_stage = '#fix1'
                    self._tf_set_ego_speed(40)

                # Desc: 前车在前方30-40米
                elif 40 >= distance > 30:
                    ego_stage = '#dynamic1'
                    self._tf_set_ego_speed(30)

                # Desc: 前车在前方25-30米
                elif 30 >= distance > 25:
                    # self.carla_desc.freeze()
                    if right_car_num == 0:
                        ego_stage = '#dynamic2_1'
                    else:
                        ego_stage = '#dynamic2_2'
                    self._tf_set_ego_speed(20)

                # Desc: 前车在前方25米内
                elif distance <= 25:
                    if not self.lane_changed:
                        self.traffic_manager.force_lane_change(ego, False)
                        print(f'\n#################### force lane change to left')
                        self.lane_changed = True
                    if right_car_num == 0:
                        ego_stage = '#dynamic3_1'
                    else:
                        ego_stage = '#dynamic3_2'
                    self._tf_set_ego_speed(20)

                # Desc: 距离异常
                else:
                    print(f'Error: distance = {distance}m')
                    raise NotImplementedError
            else:
                # Desc: 还未完成变道
                if cur_ego_loc.distance(cur_wp.transform.location) > cur_wp.lane_width / 4.0:
                    if right_car_num == 0:
                        ego_stage = '#dynamic4_1'
                    else:
                        ego_stage = '#dynamic4_2'
                    self._tf_set_ego_speed(20)

                # Desc: 已经完成变道
                else:
                    # self.carla_desc.unfreeze()
                    if ego_speed < 30:
                        ego_stage = '#fix2_2'
                    else:
                        ego_stage = '#fix2_2'
                    self._tf_set_ego_speed(40)
        else:
            self._tf_set_ego_speed(40)
            ego_stage = '#fix3'

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage.split('_')[0]
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = 'follow lane'
        explainable_data['info'] = info

        meta_input = {'sce_name': self.name, 'chuyang_inte': explainable_data}
        reason = _generate_scenario_desc_balance(meta_input)['description_demo_eng']

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{env_description}'
        # hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        # info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')
        # print(info)

        return explainable_data
        
    def _initialize_actors(self, config):
        # Depend: 场景保证左侧存在可行驶车道
        lane_info = get_lane_info(self.starting_wp)

        # For: 生成NPC
        nearby_spawn_points = get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 5))

        # For: 在自车前方50米生成一辆车
        _first_vehicle_wp = move_waypoint_forward(self.starting_wp, self.front_car_distance)
        front_actor = CarlaDataProvider.request_new_actor('*vehicle*', _first_vehicle_wp.transform)
        self.other_actors.append(front_actor)
        self.actor_desc.append('front')
        self.front_index = len(self.other_actors) - 1

        # For: 右侧如果存在车道，80%的概率生成车流
        if lane_info.r2l > 1 and random.random() < 0.9:
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
        for index, (a_desc, actor) in enumerate(zip(self.actor_desc, self.other_actors)):
            actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
            if 'npc' in a_desc:
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))
                self.traffic_manager.auto_lane_change(actor, False)
                index -= 1
            else:
                if index == 0:
                    self.traffic_manager.set_desired_speed(actor, 10)
                else:
                    self.traffic_manager.set_desired_speed(actor, 15)
    
    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="RoadLeftLaneChangeOne", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="RoadLeftLaneChangeOne_c1")
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 10))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 30, comparison_operator=operator.gt))
        root.add_child(c1)
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 20))
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
        # self.remove_all_actors()
        pass

    