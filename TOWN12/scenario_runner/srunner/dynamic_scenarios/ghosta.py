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
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy, SwitchWrongDirectionTest, BasicAgentBehavior, ScenarioTimeout, Idle, WaitForever, HandBrakeVehicle, OppositeActorFlow, GhostKeepVelocity)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, WaitEndIntersection, InTriggerDistanceToVehicle, InTriggerDistanceToNextIntersection, InTimeToArrivalToLocation, InTriggerDistanceToLocation)
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior
from TOWN12.scenario_runner.srunner.dynamic_scenarios.functions.record_data import RecordData

from TOWN12.town12_tools.explainable_utils import *
from .functions import *
from .basic_desc import *

# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class GhostA(BasicScenario):
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

        # For: original parameters
        self.first_vehicle_distance = 14.75
        self.second_vehicle_distance = 23.5
        self.pedestrian_distance = 19.25
        self._num_lane_changes = 0
        self._adversary_type = 'walker.*'
        self._adversary_speed = 3.0
        self._reaction_time = 0.8
        self._reaction_ratio = 0.12
        self._min_trigger_dist = 6.0

        # For: tick autopilot
        self.front_car = None
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        self.pedes_index = -1
        self.is_stopped = False
        self.last_speed = 0
        self.pedestrians = []
        
        # For: Is success
        self.passed_lane_ids = []
        
        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], STOP, [], ''),
            '#fix3': (KEEP_L, [], STOP, [], ''),
            '#fix4': (KEEP_L, [], ACC, [], ''),
            '#fix5': (KEEP_L, [], KEEP_S, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("GhostA", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

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
        distance = self.other_actors[self.pedes_index].get_location().distance(cur_ego_loc)
        azimuth_angle = calc_relative_position(self.ego_vehicles[0], self.other_actors[self.pedes_index], only_azimuth=True)

        RecordData.record_data(ego, cur_wp, self._world, self.other_actors)

        global last_run_time
        global last_output
        #
        current_time = time.time()
        # # 检查是否已经过了60秒
        if current_time - last_run_time >= 30:
            # 更新上次运行的时间
            last_run_time = current_time
            print("LLM starts!!!")
            RecordData._monitor_loop()
            time.sleep(2)

        info = f'Speed: {int(ego_speed)} km/h Distance: {round(distance, 2)} m'

        # Desc: 在前方
        if azimuth_angle < 90 or azimuth_angle > 270:
            # Desc: 距离较远，看不见
            if distance > 10:
                ego_stage = '#fix1'
                reason = 'Because there are no special circumstances, normal driving.'
                self._tf_set_ego_speed(20)
            else:
                # Desc: 在右前方，且距离较近，停车
                if azimuth_angle < 90:
                    ego_stage = '#fix2'
                    reason = 'Because there is a pedestrian crossing the road ahead, stop.'
                    if distance < 7:
                        self._tf_set_ego_speed(0)
                        self.is_stopped = True
                else:
                    pedes_wp = self._map.get_waypoint(self.other_actors[self.pedes_index].get_location())
                    # Desc: 在左前方，且距离较近，停车
                    if pedes_wp.lane_id == cur_wp.lane_id:
                        ego_stage = '#fix3'
                        reason = 'Because there is a pedestrian crossing the road ahead, stop.'
                        self._tf_set_ego_speed(0)
                        self.is_stopped = True
                    # Desc: 在左前方，且距离较近，但是在另一条车道上，正常行驶
                    else:
                        if ego_speed < 15:
                            ego_stage = '#fix4'
                            reason = 'Because the pedestrian has crossed the road, accelerate.'
                            self._tf_set_ego_speed(20)
                        else:
                            ego_stage = '#fix5'
                            reason = 'Because there are no special circumstances, normal driving.'
                            self._tf_set_ego_speed(20)
        else:
            # Desc: 在后方，正常行驶
            ego_stage = '#fix6'
            reason = 'Because there are no special circumstances, normal driving.'
            self._tf_set_ego_speed(20)

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description(scenario='ghosta')

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
        # print(f'\r{info}', end='\n')

        self.last_speed = ego_speed

        return explainable_data
        
    def _initialize_actors(self, config):
        lane_info = get_lane_info(self.starting_wp)

        nearby_spawn_points = get_different_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(5, 30), 60, allow_same_side=True, allow_behind=True)
        nearby_ped_spawn_points = get_random_pedestrian_transforms(self._world, self.ego_vehicles[0], random.randint(5, 10))

        waypoint = self.starting_wp
        wp_next = waypoint.get_right_lane()

        self.wp_first_vehicle, _ = get_waypoint_in_distance(wp_next, self.first_vehicle_distance)
        self.wp_second_vehicle, _ = get_waypoint_in_distance(wp_next, self.second_vehicle_distance)
        self.wp_pedestrian, _ = get_waypoint_in_distance(wp_next, self.pedestrian_distance)
        self._collision_wp, _ = get_waypoint_in_distance(self.starting_wp, self.pedestrian_distance)

        sidewalk_waypoint = self.wp_pedestrian
        while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
            right_wp = sidewalk_waypoint.get_right_lane()
            if right_wp is None:
                break
            sidewalk_waypoint = right_wp
            self._num_lane_changes += 1

        offset = {'yaw': 270, 'z': 0.5, 'k': 1.0}
        self._adversary_transform = get_sidewalk_transform(sidewalk_waypoint, offset)

        adversary = CarlaDataProvider.request_new_actor(self._adversary_type, self._adversary_transform)
        adversary.set_simulate_physics(enabled=True)
        self.other_actors.append(adversary)
        self.actor_desc.append('ghost_pedestrian')
        self.pedes_index = len(self.other_actors) - 1

        # For: 100%的概率在后面生成一辆车
        if np.random.rand() < 1:
            first_vehicle_temp_transform = carla.Transform(
                carla.Location(
                    self.wp_first_vehicle.transform.location.x,
                    self.wp_first_vehicle.transform.location.y,
                    self.wp_first_vehicle.transform.location.z),
                self.wp_first_vehicle.transform.rotation
            )
            bp_name = choice(LARGE_VEHICLE)
            first_vehicle = CarlaDataProvider.request_new_actor(bp_name, first_vehicle_temp_transform)
            self.other_actors.append(first_vehicle)
            self.actor_desc.append('ghost_vehicle1')

        # For: 60%的概率在前面生成一辆车
        if np.random.rand() < 0.6:
            second_vehicle_temp_transform = carla.Transform(
                carla.Location(
                    self.wp_second_vehicle.transform.location.x,
                    self.wp_second_vehicle.transform.location.y,
                    self.wp_second_vehicle.transform.location.z),
                self.wp_second_vehicle.transform.rotation)
            bp_name = choice(LARGE_VEHICLE)
            second_vehicle = CarlaDataProvider.request_new_actor(bp_name, second_vehicle_temp_transform)
            self.other_actors.append(second_vehicle)
            self.actor_desc.append('ghost_vehicle2')

        # For: 生成NPC
        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        # For: 生成行人
        count = 0
        for p_index, (p_bp, p_transform, p_dest_loc) in enumerate(nearby_ped_spawn_points):
            pedestrian, controller = gen_ai_walker(self._world, p_transform, CarlaDataProvider)
            if pedestrian is not None:
                controller.start()
                controller.go_to_location(p_dest_loc)
                controller.set_max_speed(random.randint(10, 30) / 10.0)
                self.pedestrians.append([pedestrian, controller])
                count += 1

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_desc, actor in zip(self.actor_desc, self.other_actors):
            if actor.type_id.startswith('vehicle'):
                if not a_desc.startswith('ghost'):
                    actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                    self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))
                    self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        sequence = py_trees.composites.Sequence()
        collision_location = self._collision_wp.transform.location
        collision_distance = collision_location.distance(self._adversary_transform.location)
        collision_duration = collision_distance / self._adversary_speed
        reaction_time = self._reaction_time - self._reaction_ratio * self._num_lane_changes
        collision_time_trigger = collision_duration + reaction_time

        trigger_adversary = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")

        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], collision_time_trigger, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)

        # Move the adversary
        speed_duration = 2.0 * collision_duration
        speed_distance = 2.0 * collision_distance
        sequence.add_child(GhostKeepVelocity(
            self.other_actors[0], self._adversary_speed,
            duration=speed_duration, distance=speed_distance, name="AdversaryCrossing")
        )
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryEnd")
        end_condition.add_child(InTriggerDistanceToVehicle(self.other_actors[self.pedes_index], self.ego_vehicles[0], 15, comparison_operator=operator.gt))
        end_condition.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 8.5))
        end_condition.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        sequence.add_child(end_condition)

        return sequence
    
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
