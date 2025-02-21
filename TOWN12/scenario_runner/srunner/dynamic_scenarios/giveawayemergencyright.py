from __future__ import print_function

import time

# import ipdb
import numpy as np
import py_trees
import carla
import time

import re
import operator
from random import choice
from agents.navigation.local_planner import RoadOption

from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy, SwitchWrongDirectionTest, BasicAgentBehavior, ScenarioTimeout, Idle, WaitForever, HandBrakeVehicle, OppositeActorFlow)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, WaitEndIntersection, InTriggerDistanceToVehicle, InTriggerDistanceToNextIntersection, InTriggerDistanceToLocation, ScenarioNPCTriggerFunction)
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior
from TOWN12.scenario_runner.srunner.dynamic_scenarios.functions.record_data import RecordData

from TOWN12.town12_tools.explainable_utils import *
from .functions import *
from .basic_desc import *

# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class GiveAwayEmergencyRight(BasicScenario):
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

        # For: Other actors parameters
        self.emergency_behind_distance = 45
        self.parallel_behind_distance = 10

        # For: tick autopilot
        self.emergency_actor = None
        self.parallel_actor = None
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 30
        self.navigation_cmds = ['Straight']
        self.emergency_index = -1
        
        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#dynamic1': (RIGHT_C, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], KEEP_S, [], ''),
            '#fix3': (KEEP_L, [], KEEP_S, [], ''),
        }

        self.debug_time = time.time()

        super().__init__("GiveAwayEmergencyRight", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'
        change_num = 0
        lane_id = self.passed_lane_ids[0]
        for item in self.passed_lane_ids[1:]:
            if item != lane_id:
                change_num += 1
                lane_id = item
        if change_num != 1:
            return False, f'Autopilot变道次数异常，应为1次，实际为{change_num}次'
        return True, ''

    def uniad_tick_autopilot(self, max_speed):
        ego = self.ego_vehicles[0]

        if self.emergency_actor is None:
            self.emergency_actor = self.other_actors[self.actor_desc.index('emergency')]
        emergency_actor_wp = self._map.get_waypoint(self.emergency_actor.get_location())
        emergency_actor_loc = self.emergency_actor.get_location()

        if self.parallel_actor is None:
            self.parallel_actor = self.other_actors[self.actor_desc.index('parallel')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(max_speed)
            self._set_ego_autopilot()
            self.initialized = True

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        if self.emergency_actor.is_alive and self.parallel_actor.is_alive:
            em_distance = cur_ego_loc.distance(emergency_actor_loc)

            # Desc: 自车和紧急车辆在同一车道
            if cur_wp.lane_id == emergency_actor_wp.lane_id:
                if em_distance > 30:
                    self._tf_set_ego_speed(max_speed)
                else:
                    if not self.lane_changed and em_distance < 25:
                        self._force_ego_lanechange_right()
                        self.lane_changed = True
                        print(f'\n\nChange lane to right')
                    self._tf_set_ego_speed(20)
            else:
                self._tf_set_ego_speed(max_speed)
        else:
            self._tf_set_ego_speed(max_speed)

        return {'cur_wp': cur_wp, 'navi_cmd': 'follow lane'}

    def _get_navi_route(self, cur_wp):
        if cur_wp.lane_id == self.starting_wp.lane_id:
            start_wp = cur_wp
        else:
            start_wp = cur_wp.get_left_lane()
        dest_wp = self.starting_wp.next_until_lane_end(1)[-2].next(150)[0]
        return start_wp, dest_wp

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.emergency_actor is None:
            self.emergency_actor = self.other_actors[self.actor_desc.index('emergency')]
        emergency_actor_wp = self._map.get_waypoint(self.emergency_actor.get_location())
        emergency_actor_loc = self.emergency_actor.get_location()

        if self.parallel_actor is None:
            self.parallel_actor = self.other_actors[self.actor_desc.index('parallel')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._set_ego_autopilot()
            self.initialized = True

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors, eng=True),
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
        if current_time - last_run_time >= 10:
            # 更新上次运行的时间
            last_run_time = current_time
            print("LLM starts!!!")
            RecordData._monitor_loop()
            time.sleep(2)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        info = f'Speed: {int(ego_speed)} km/h '

        if self.emergency_actor.is_alive and self.parallel_actor.is_alive:
            em_desc = get_description(explainable_data, self.emergency_actor)
            pa_desc = get_description(explainable_data, self.parallel_actor)
            em_distance = cur_ego_loc.distance(emergency_actor_loc)
            info += f'Distance: {int(em_distance)}m '

            # Desc: 自车和紧急车辆在同一车道
            if cur_wp.lane_id == emergency_actor_wp.lane_id:
                if em_distance > 40:
                    ego_stage = '#fix1'
                    reason = 'Because there are no special circumstances, normal driving.'
                    self._tf_set_ego_speed(30)
                else:
                    if not self.lane_changed and em_distance < 25:
                        self._force_ego_lanechange_right()
                        self.lane_changed = True

                    ego_stage = '#dynamic1'
                    reason = f'Because a {em_desc.color} {em_desc.type} is approaching from behind，and there are multiple vehicles on the left，change lanes to the right to yield to the {em_desc.color} {em_desc.type}.'
                    self._tf_set_ego_speed(30)
            else:
                ego_stage = '#fix2'
                reason = 'Because there are no special circumstances, normal driving.'
                self._tf_set_ego_speed(30)
        else:
            ego_stage = '#fix3'
            reason = 'Because there are no special circumstances, normal driving.'
            self._tf_set_ego_speed(30)

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

        info += f'{ego_stage} -> 可解释性描述：{reason}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        lane_info = get_lane_info(self.starting_wp)
        emergency_vehicle_names = ['vehicle.ford.ambulance', 'vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020']

        # For: 生成NPC
        nearby_spawn_points = get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 5))

        # For: 在自车后方45米生成一辆车
        _first_vehicle_wp = move_waypoint_backward(self.starting_wp, self.emergency_behind_distance)
        vehicle_model_name = choice(emergency_vehicle_names)
        emergency_actor = CarlaDataProvider.request_new_actor(vehicle_model_name, _first_vehicle_wp.transform)
        self.other_actors.append(emergency_actor)
        self.actor_desc.append('emergency')
        self.emergency_index = len(self.other_actors) - 1

        # For: 在自车右侧后方10米生成一辆车（并排行驶）
        _second_vehicle_wp = move_waypoint_backward(self.starting_wp.get_right_lane(), self.parallel_behind_distance)
        vehicle_model_name = get_blueprint(self._world, '*vehicle*', excludes=emergency_vehicle_names)
        parallel_actor = CarlaDataProvider.request_new_actor(vehicle_model_name, _second_vehicle_wp.transform)
        self.other_actors.append(parallel_actor)
        self.actor_desc.append('parallel')

        # For: 左侧如果存在车道，生成车流
        if lane_info.l2r > 1:
            left_vehicles = left_lane_vehicle_flow_scenario(self.starting_wp)
            for v_index, (v_bp, v_transform) in enumerate(left_vehicles):
                left_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
                self.other_actors.append(left_actor)
                self.actor_desc.append(f'left_{v_index}')

        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()

    def start_npc(self):
        for desc, actor in zip(self.actor_desc, self.other_actors):
            if actor.type_id.startswith('vehicle'):
                if desc == 'emergency':
                    # self.traffic_manager.ignore_vehicles_percentage(actor, 100)
                    self.traffic_manager.set_desired_speed(actor, 100)
                    self.traffic_manager.ignore_lights_percentage(actor, 100)
                    self.traffic_manager.ignore_signs_percentage(actor, 100)
                else:
                    self.traffic_manager.set_desired_speed(actor, 30)
                self.traffic_manager.auto_lane_change(actor, False)
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())

    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="GiveAwayEmergencyRight")
        start_condition = InTriggerDistanceToLocation(self.ego_vehicles[0], self.starting_wp.transform.location, self.starting_wp.lane_width)
        root.add_child(start_condition)
        npc_function = ScenarioNPCTriggerFunction(self)
        root.add_child(npc_function)
        end_condition = py_trees.composites.Parallel(name="GiveAwayEmergencyRight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="GiveAwayEmergencyRight_c1")
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.emergency_index], self.ego_vehicles[0], 10))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.emergency_index], self.ego_vehicles[0], 40, comparison_operator=operator.gt))
        end_condition.add_child(c1)
        end_condition.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
        root.add_child(end_condition)
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

    