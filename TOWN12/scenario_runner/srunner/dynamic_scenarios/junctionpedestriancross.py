from __future__ import print_function

import random

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
# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class JunctionPedestrianCross(BasicScenario):
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
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 30
        self.navigation_cmds = []
        self.training_nav = []
        self.pedestrians = []
        self.is_stopped = False

        # Desc: 判断当前车道能否直行、左转、右转
        junction = get_next_junction(self.starting_wp)
        if get_junction_turn_after_wp(junction, self.starting_wp, 'right', None) is not None:
            self.navigation_cmds.append('Right')
            self.training_nav.append('turn right')
        if get_junction_turn_after_wp(junction, self.starting_wp, 'left', None) is not None:
            self.navigation_cmds.append('Left')
            self.training_nav.append('turn left')
        if get_junction_turn_after_wp(junction, self.starting_wp, 'straight', None) is not None:
            self.navigation_cmds.append('Straight')
            self.training_nav.append('follow lane')

        # Desc: 随机选择一个导航命令
        random_nav_index = random.randint(0, len(self.navigation_cmds) - 1)
        self.navigation_cmds = [self.navigation_cmds[random_nav_index]]
        self.training_nav = self.training_nav[random_nav_index]

        # For: Is success
        self.passed_lane_ids = []
        
        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], STOP, [], ''),
            '#fix3': (KEEP_L, [], DEC, [], ''),
            '#fix4': (KEEP_L, [], STOP, [], ''),
            '#fix5': (KEEP_L, [], DEC, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("JunctionPedestrianCross", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        if not self.is_stopped:
            return False, 'ego vehicle not stopped'
        return True, ''

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

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
        if current_time - last_run_time >= 10:
            # 更新上次运行的时间
            last_run_time = current_time
            print("LLM starts!!!")
            RecordData._monitor_loop()
            time.sleep(2)

        info = f'Speed: {int(ego_speed)} km/h '
        distance = distance_to_next_junction(cur_wp)
        info += f'Distance: {distance}m '

        if ego_speed < 0.1 and cur_ego_loc.distance(self.starting_wp.transform.location) < 1:
            ego_stage = '#fix1'
            reason = 'Because there are no special circumstances, normal driving.'
            self._tf_set_ego_speed(30)
        else:
            ped_in_front = has_pedestrian_in_front(self.pedestrians, ego, distance=distance, lane_width=cur_wp.lane_width)
            if ped_in_front:
                if distance < 10:
                    ego_stage = '#fix2'
                    if len(self.pedestrians) > 1:
                        reason = f'Because there are pedestrians crossing in front of ego, stop and wait for pedestrians to pass.'
                    else:
                        reason = f'Because there is a pedestrian crossing in front of ego, stop and wait for pedestrians to pass.'
                    self._tf_set_ego_speed(0)
                    self.is_stopped = True
                else:
                    ego_stage = '#fix3'
                    if len(self.pedestrians) > 1:
                        reason = 'Because there are pedestrians crossing at the intersection far ahead, decelerate.'
                    else:
                        reason = f'Because there is a pedestrian crossing at the intersection far ahead, decelerate.'
                    self._tf_set_ego_speed(15)
            else:
                if ego_speed < 0.1:
                    if junction_has_traffic_light(self._world, cur_wp):
                        ego_stage = '#fix4'
                        reason = 'Because of the red light at intersection, stop and wait.'
                        self._tf_set_ego_speed(30)
                    else:
                        ego_stage = '#fix5'
                        reason = 'Because it is close to intersection, slow down and observe.'
                        self._tf_set_ego_speed(30)
                else:
                    ego_stage = '#fix6'
                    reason = 'Because there are no special circumstances, normal driving.'
                    self._tf_set_ego_speed(30)

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = self.training_nav
        explainable_data['info'] = info

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{reason}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()

        nearby_spawn_points = get_different_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(5, 30), 60, allow_same_side=True, allow_behind=True)

        pedestrian_end_wp, _ = get_sidewalk_wps(self.starting_wp)
        pedestrian_start_wp = self.starting_wp.next_until_lane_end(1)[-1]
        _, pedestrian_start_wp = get_sidewalk_wps(pedestrian_start_wp)

        for i in range(random.randint(10, 15)):
            pedestrian, controller = gen_ai_walker(self._world, pedestrian_start_wp.transform, CarlaDataProvider)
            controller.start()
            controller.go_to_location(pedestrian_end_wp.transform.location)
            controller.set_max_speed(random.randint(10, 30) / 10.0)
            self.pedestrians.append([pedestrian, controller])
            pedestrian_start_wp = pedestrian_start_wp.previous(1.0)[0]

        # For: 生成NPC
        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        for actor in self.other_actors:
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))
                self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="JunctionPedestrianCross", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 0.2))
        root.add_child(DriveDistance(self.ego_vehicles[0], 50))
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

    