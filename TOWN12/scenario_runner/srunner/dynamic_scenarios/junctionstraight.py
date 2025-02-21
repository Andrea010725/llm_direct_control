from __future__ import print_function

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

class JunctionStraight(BasicScenario):
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
        self.initialized = False
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        
        # For: Is success
        self.passed_lane_ids = []
        
        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], STOP, [], ''),
            '#fix2': (KEEP_L, [], DEC, [], ''),
            '#fix3': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("JunctionStraight", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        return True, ''

    def uniad_tick_autopilot(self, max_speed):
        ego = self.ego_vehicles[0]
        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(max_speed)
            self._set_ego_autopilot()
            self.initialized = True

        self._tf_set_ego_speed(max_speed)

        return {'cur_wp': cur_wp, 'navi_cmd': 'follow lane'}

    def _get_navi_route(self, cur_wp):
        start_wp = cur_wp
        dest_wp = get_junction_turn_after_wp(get_next_junction(self.starting_wp), self.starting_wp, 'straight', None).next(5)[0].next(70)[0]
        return start_wp, dest_wp

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._set_ego_autopilot()
            self.initialized = True

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

        explainable_data = {}
        info = f'Speed: {int(ego_speed)} km/h '

        # Desc: 速度小于0且距离起点大于1m
        if ego_speed < 0.1 and self.starting_wp.transform.location.distance(cur_ego_loc) > 1:
            # Desc: 路口是有红绿灯的 -> 速度为0，说明红灯
            if junction_has_traffic_light(self._world, cur_wp):
                ego_stage = '#fix1'
                reason = 'because the traffic light at the intersection ahead is red, stop and wait.'
                self._tf_set_ego_speed(20)
            # Desc: 路口是无红绿灯的 -> 速度为0，说明STOP标志
            else:
                ego_stage = '#fix2'
                reason = 'because of the approaching junction, drive slowly.'
                self._tf_set_ego_speed(20)
        else:
            ego_stage = '#fix3'
            reason = 'Because there are no special circumstances, normal driving.'
            self._tf_set_ego_speed(20)
        
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
        print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        nearby_spawn_points = get_different_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(5, 30), 60, allow_same_side=True, allow_behind=False)
        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            self.other_actors.append(right_actor)
            self.actor_desc.append(f'npc_{v_index}')

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for actor in self.other_actors:
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))

        # self.traffic_manager.ignore_lights_percentage(self.ego_vehicles[0], 100.0) # WENYANG ADD 闯红灯 20231115
    
    def _create_behavior(self):
        root = py_trees.composites.Sequence(name="JunctionStraight")
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))
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

    