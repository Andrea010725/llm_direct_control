#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

# import ipdb
import numpy as np
import py_trees
import carla

from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      SwitchWrongDirectionTest,
                                                                      BasicAgentBehavior,
                                                                      ScenarioTimeout,
                                                                      Idle, WaitForever,
                                                                      HandBrakeVehicle,
                                                                      OppositeActorFlow)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle,
                                                                               WaitUntilInFront,
                                                                               WaitUntilInFrontPosition)
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior

from TOWN12.town12_tools.explainable_utils import *


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default


class Accident(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to lane change. A police vehicle is located before
    two other cars that have been in an accident.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        
        self._first_distance = 10
        self._second_distance = 6

        self._trigger_distance = 30
        self._end_distance = 50
        self._wait_duration = 5
        self._offset = 0.6

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2 | carla.VehicleLightState.Position

        self._distance = get_value_parameter(config, 'distance', float, 80)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        CarlaDataProvider._accident_direction = self._direction

        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        super().__init__(
            "Accident", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_side_prop(self, wp):
        # Spawn the accident indication signal
        prop_wp = wp
        while True:
            if self._direction == "right":
                wp = prop_wp.get_right_lane()
            else:
                wp = prop_wp.get_left_lane()
            if wp is None or wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                break
            prop_wp = wp

        displacement = 0.3 * prop_wp.lane_width
        r_vec = prop_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=0.2)
        spawn_transform.rotation.yaw += 90
        signal_prop = CarlaDataProvider.request_new_actor('static.prop.warningaccident', spawn_transform)
        if not signal_prop:
            raise ValueError("Couldn't spawn the indication prop asset")
        signal_prop.set_simulate_physics(True)
        self.other_actors.append(signal_prop)

    def _spawn_obstacle(self, wp, blueprint, accident_actor=False):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        if accident_actor:
            actor = CarlaDataProvider.request_new_actor(
                blueprint, spawn_transform, rolename='scenario no lights', attribute_filter={'base_type': 'car', 'generation': 2})
        else:
            actor = CarlaDataProvider.request_new_actor(
                blueprint, spawn_transform, rolename='scenario')
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        police_car = self.other_actors[1]
        first_car = self.other_actors[2]
        second_car = self.other_actors[3]

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors),
        }

        if police_car.is_alive and first_car.is_alive and second_car.is_alive:
            police_desc = Edict(explainable_data['actors'][police_car.id]['description'])
            police_data = Edict(explainable_data['actors'][police_car.id])

            first_desc = Edict(explainable_data['actors'][first_car.id]['description'])
            first_car_data = Edict(explainable_data['actors'][first_car.id])

            second_desc = Edict(explainable_data['actors'][second_car.id]['description'])
            second_car_data = Edict(explainable_data['actors'][second_car.id])
            print(f'Distance: {police_data.distance}  Speed: {ego_speed} km/h')

            # 距离警车50米内，且警车在前方，减速准备变道
            if 40 < police_data.distance < 50 and police_data.cos_value > 0.1:
                self._tf_set_ego_speed(20)
                ego_stage = '#dynamic1'
                ego_action = 'decelerate'
                explainable_desc = f'{police_desc.direction}{police_desc.distance}处有一辆{police_desc.color}{police_desc.type}{police_desc.speed}，减速'
            elif police_data.distance <= 40 and police_data.cos_value > 0.1 and get_actor_lane_id(self.world, police_car) == get_actor_lane_id(self.world, ego):
                self._tf_set_ego_speed(10)
                ego_stage = '#dynamic2'
                ego_action = 'decelerate|lanechange|left'
                explainable_desc = f'{police_desc.direction}{police_desc.distance}处有一辆{police_desc.color}{police_desc.type}{police_desc.speed}，减速准备向左变道'
            elif get_actor_lane_id(self.world, police_car) != get_actor_lane_id(self.world, ego):
                # 在旁边车道，低速通过
                if second_car_data.cos_value > 0.1:
                    self._tf_set_ego_speed(20)
                    ego_stage = '#dynamic3'
                    ego_action = 'hold|lowspeed'
                    explainable_desc = f'{second_desc.direction}{second_desc.distance}处一辆{second_desc.color}{second_desc.type}与一辆{first_desc.color}{first_desc.type}发生交通事故，低速通过'
                else:
                    self._tf_set_ego_speed(10)
                    ego_stage = '#fix2'
                    ego_action = 'decelerate|lanechange|right'
                    explainable_desc = f'通过右侧交通事故路段，向右变道回到原车道'
                    # 已经变道回来了且距离警车70米外，加速
            elif police_data.cos_value < 0.1 and get_actor_lane_id(self.world, police_car) == get_actor_lane_id(self.world, ego) and police_data.distance > 55:
                self._tf_set_ego_speed(40)
                ego_stage = '#fix3'
                ego_action = 'accelerate'
                explainable_desc = f'恢复正常行驶'
            elif police_data.cos_value < 0.1 and get_actor_lane_id(self.world, police_car):
                self._tf_set_ego_speed(10)
                ego_stage = '#fix4'
                ego_action = 'hold|lowspeed'
                explainable_desc = f'完成变道，准备加速'
            # 已经变道回来了且距离警车70米外，加速
            else:
                self._tf_set_ego_speed(40)
                ego_stage = '#fix1'
                ego_action = 'hold|normalspeed'
                explainable_desc = f'正常行驶'
        else:
            print(f'Distance: NaN  Speed: {ego_speed} km/h  Desc: Pass Scenario -> High Speed')
            self._tf_set_ego_speed(40)
            ego_stage = '#fix5'
            ego_action = 'hold|normalspeed'
            explainable_desc = f'正常行驶'

        explainable_data['explainable_desc'] = explainable_desc
        explainable_data['ego_stage'] = ego_stage
        explainable_data['scenario'] = 'Accident'
        explainable_data['ego_action'] = ego_action
        print(f'可解释性描述：{explainable_desc}')

        return explainable_data

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Spawn the accident indication signal
        self._spawn_side_prop(starting_wp)

        # Spawn the police vehicle
        self._accident_wp = self._move_waypoint_forward(starting_wp, self._distance)
        police_car = self._spawn_obstacle(self._accident_wp, 'vehicle.dodge.charger_police_2020')

        # Set its initial conditions
        lights = police_car.get_light_state()
        lights |= self._lights
        police_car.set_light_state(carla.VehicleLightState(lights))
        police_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        self._first_vehicle_wp = self._move_waypoint_forward(self._accident_wp, self._first_distance)
        first_actor = self._spawn_obstacle(self._first_vehicle_wp, 'vehicle.*', True)

        # Set its initial conditions
        first_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(first_actor)

        # Create the second vehicle that has been in the accident
        second_vehicle_wp = self._move_waypoint_forward(self._first_vehicle_wp, self._second_distance)
        second_actor = self._spawn_obstacle(second_vehicle_wp, 'vehicle.*', True)

        self._accident_wp = second_vehicle_wp
        self._end_wp = self._move_waypoint_forward(second_vehicle_wp, self._end_distance)
        scenario_end_wp = self._move_waypoint_forward(second_vehicle_wp, 3)
        CarlaDataProvider._scenario_end_wp = scenario_end_wp

        # Set its initial conditions
        second_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(second_actor)

    def _create_behavior(self):
        """
        The vehicle has to drive the reach a specific point but an accident is in the middle of the road,
        blocking its route and forcing it to lane change.
        """
        root = py_trees.composites.Sequence(name="Accident")
        if self.route_mode:
            total_dist = self._distance + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._first_vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

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
        self.remove_all_actors()


class AccidentTwoWays(Accident):
    """
    Variation of the Accident scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [40, 105])
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        police_car = self.other_actors[1]
        first_car = self.other_actors[2]
        second_car = self.other_actors[3]

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors),
            'nick_names': {
                police_car.id: 'police',
                first_car.id: 'first',
                second_car.id: 'second',
                ego.id: 'ego',
            }
        }

        if police_car.is_alive and first_car.is_alive and second_car.is_alive:
            police_desc = Edict(explainable_data['actors'][police_car.id]['description'])
            police_data = Edict(explainable_data['actors'][police_car.id])

            first_desc = Edict(explainable_data['actors'][first_car.id]['description'])
            first_car_data = Edict(explainable_data['actors'][first_car.id])

            second_desc = Edict(explainable_data['actors'][second_car.id]['description'])
            second_car_data = Edict(explainable_data['actors'][second_car.id])
            print(f'\rDistance: {round(police_data.distance, 2)}  Speed: {ego_speed} km/h ', end='')

            # 距离警车50米内，且警车在前方，减速准备变道
            if police_data.distance > 50 and police_data.cos_value > 0.1:
                self._tf_set_ego_speed(40)
                ego_stage = '#fix1'
                ego_action = 'hold|normalspeed'
                explainable_desc = f'正常行驶'
            elif 40 < police_data.distance < 50 and police_data.cos_value > 0.1:
                self._tf_set_ego_speed(20)
                ego_stage = '#dynamic1'
                ego_action = 'decelerate|lowspeed'
                explainable_desc = f'{police_desc.direction}{police_desc.distance}处有一辆{police_desc.color}{police_desc.type}{police_desc.speed}，减速'
            elif 30 <= police_data.distance <= 40 and police_data.cos_value > 0.1 and get_actor_lane_id(self.world, police_car) == get_actor_lane_id(self.world, ego):
                self._tf_set_ego_speed(15)
                ego_stage = '#dynamic2'
                ego_action = 'decelerate|lowspeed|lanechange|left'
                explainable_desc = f'{police_desc.direction}{police_desc.distance}处有一辆{police_desc.color}{police_desc.type}{police_desc.speed}，减速'
            elif police_data.distance <= 30 and police_data.cos_value > 0.1:
                self._tf_set_ego_speed(15)
                lane_width = get_actor_lane_width(self.world, ego)
                self._tf_set_ego_offset(-lane_width)
                self._tf_set_ego_force_go(100)
                ego_stage = '#dynamic3'
                ego_action = 'hold|lowspeed|lanechange|left|invasion'
                explainable_desc = f'{police_desc.direction}{police_desc.distance}处有一辆{police_desc.color}{police_desc.type}{police_desc.speed}，向左借道'
            elif second_car_data.cos_value > 0.1:
                self._tf_set_ego_speed(20)
                lane_width = get_actor_lane_width(self.world, ego)
                self._tf_set_ego_offset(-lane_width)
                self._tf_set_ego_force_go(100)
                ego_stage = '#dynamic4'
                ego_action = 'hold|lowspeed|invasion'
                explainable_desc = f'{second_desc.direction}{second_desc.distance}处一辆{second_desc.color}{second_desc.type}与一辆{first_desc.color}{first_desc.type}发生交通事故，借道通过'
            elif second_car_data.distance < 10 and second_car_data.cos_value < 0.1:
                self._tf_set_ego_speed(15)
                lane_width = get_actor_lane_width(self.world, ego)
                self._tf_set_ego_offset(-lane_width)
                self._tf_set_ego_force_go(100)
                ego_stage = '#dynamic5'
                ego_action = 'decelerate|lowspeed|invasion'
                explainable_desc = f'{second_desc.direction}{second_desc.distance}处一辆{second_desc.color}{second_desc.type}与一辆{first_desc.color}{first_desc.type}发生交通事故，借道通过'
            elif second_car_data.distance >= 10 and second_car_data.cos_value < 0.1 and get_actor_lane_id(self.world, police_car, False) != get_actor_lane_id(self.world, ego, False):
                self._tf_set_ego_speed(15)
                self._tf_set_ego_offset(0)
                self._tf_set_ego_force_go(100)
                ego_stage = '#fix2'
                ego_action = 'hold|lowspeed|lanechange|right'
                explainable_desc = f'通过右侧交通事故路段，向右变道回到原车道'
            elif 20 >= second_car_data.distance >= 10 and second_car_data.cos_value < 0.1 and get_actor_lane_id(self.world, police_car) == get_actor_lane_id(self.world, ego):
                self._tf_set_ego_speed(15)
                self._tf_set_ego_offset(0)
                ego_stage = '#fix3'
                ego_action = 'lanechange|right'
                explainable_desc = f'通过右侧交通事故路段，向右变道回到原车道'
            elif second_car_data.distance >= 20 and second_car_data.cos_value < 0.1 and get_actor_lane_id(self.world, police_car) == get_actor_lane_id(self.world, ego):
                self._tf_set_ego_speed(40)
                self._tf_set_ego_offset(0)
                ego_stage = '#fix4'
                ego_action = 'accelerate|normalspeed'
                explainable_desc = f'恢复正常行驶'
            else:
                ego_stage = '#error'
                ego_action = 'unknown'
                explainable_desc = f'正常行驶'
        else:
            print(f'\rDistance: NaN  Speed: {ego_speed} km/h ', end='')
            self._tf_set_ego_speed(40)
            ego_stage = '#fix5'
            ego_action = 'hold|normalspeed'
            explainable_desc = f'正常行驶'

        explainable_data['explainable_desc'] = explainable_desc
        explainable_data['ego_stage'] = ego_stage
        explainable_data['scenario'] = 'Accident'
        explainable_data['ego_action'] = ego_action
        print(f' {ego_stage}: {explainable_desc}          ', end='')

        return explainable_data

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        reference_wp = self._accident_wp.get_left_lane()
        if not reference_wp:
            raise ValueError("Couldnt find a left lane to spawn the opposite traffic")

        root = py_trees.composites.Sequence(name="AccidentTwoWays")
        if self.route_mode:
            total_dist = self._distance + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._first_vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(active=False))
        behavior.add_child(OppositeActorFlow(reference_wp, self.ego_vehicles[0], self._opposite_interval))

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(True))
            behavior.add_child(ChangeOppositeBehavior(active=True))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

class ParkedObstacle(BasicScenario):
    """
    Scenarios in which a parked vehicle is incorrectly parked,
    forcing the ego to lane change out of the route's lane
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_distance = 30
        self._end_distance = 50
        self._wait_duration = 5
        self._offset = 0.7

        self._lights = carla.VehicleLightState.RightBlinker | carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.Position

        self._distance = get_value_parameter(config, 'distance', float, 120)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        CarlaDataProvider._accident_direction = self._direction

        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        super().__init__(
            "ParkedObstacle", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_side_prop(self, wp):
        # Spawn the accident indication signal
        prop_wp = wp
        while True:
            if self._direction == "right":
                wp = prop_wp.get_right_lane()
            else:
                wp = prop_wp.get_left_lane()
            if wp is None or wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                break
            prop_wp = wp

        displacement = 0.3 * prop_wp.lane_width
        r_vec = prop_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=0.2)
        spawn_transform.rotation.yaw += 90
        signal_prop = CarlaDataProvider.request_new_actor('static.prop.warningaccident', spawn_transform)
        if not signal_prop:
            raise ValueError("Couldn't spawn the indication prop asset")
        signal_prop.set_simulate_physics(True)
        self.other_actors.append(signal_prop)

    def _spawn_obstacle(self, wp, blueprint):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        actor = CarlaDataProvider.request_new_actor(
            blueprint, spawn_transform, rolename='scenario no lights', attribute_filter={'base_type': 'car', 'generation': 2})
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Create the side prop
        self._spawn_side_prop(self._starting_wp)

        # Create the first vehicle that has been in the accident
        self._vehicle_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        parked_actor = self._spawn_obstacle(self._vehicle_wp, 'vehicle.*')

        lights = parked_actor.get_light_state()
        lights |= self._lights
        parked_actor.set_light_state(carla.VehicleLightState(lights))
        parked_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(parked_actor)

        self._end_wp = self._move_waypoint_forward(self._vehicle_wp, self._end_distance)
        scenario_end_wp = self._move_waypoint_forward(self._vehicle_wp, 12)
        CarlaDataProvider._scenario_end_wp = scenario_end_wp
    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence(name="ParkedObstacle")
        if self.route_mode:
            total_dist = self._distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

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
        self.remove_all_actors()


class ParkedObstacleTwoWays(ParkedObstacle):
    """
    Variation of the ParkedObstacle scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [20, 100])
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        reference_wp = self._vehicle_wp.get_left_lane()
        if not reference_wp:
            raise ValueError("Couldnt find a left lane to spawn the opposite traffic")

        root = py_trees.composites.Sequence(name="ParkedObstacleTwoWays")
        if self.route_mode:
            total_dist = self._distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(active=False))
        behavior.add_child(OppositeActorFlow(reference_wp, self.ego_vehicles[0], self._opposite_interval))

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root


class HazardAtSideLane(BasicScenario):
    """
    Added the dangerous scene of ego vehicles driving on roads without sidewalks,
    with three bicycles encroaching on some roads in front.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._obstacle_distance = 9
        self._trigger_distance = 30
        self._end_distance = 40

        self._offset = 0.55
        self._wait_duration = 5

        self._target_locs = []

        self._bicycle_bps = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]

        self._distance = get_value_parameter(config, 'distance', float, 100)
        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._bicycle_speed = get_value_parameter(config, 'bicycle_speed', float, 10)
        self._bicycle_drive_distance = get_value_parameter(config, 'bicycle_drive_distance', float, 50)
        self._scenario_timeout = 240

        super().__init__("HazardAtSideLane",
                         ego_vehicles,
                         config,
                         world,
                         randomize,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_obstacle(self, wp, blueprint):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        actor = CarlaDataProvider.request_new_actor(blueprint, spawn_transform)
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        rng = CarlaDataProvider.get_random_seed()
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        CarlaDataProvider._scenario_start_wp = self._move_waypoint_forward(self._starting_wp, self._distance)

        # Spawn the first bicycle
        first_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        bicycle_1 = self._spawn_obstacle(first_wp, rng.choice(self._bicycle_bps))

        wps = first_wp.next(self._bicycle_drive_distance)
        if not wps:
            raise ValueError("Couldn't find an end location for the bicycles")
        self._target_locs.append(wps[0].transform.location)

        # Set its initial conditions
        bicycle_1.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(bicycle_1)

        # Spawn the second bicycle
        second_wp = self._move_waypoint_forward(first_wp, self._obstacle_distance)
        bicycle_2 = self._spawn_obstacle(second_wp, rng.choice(self._bicycle_bps))
        
        scenario_end_wp = self._move_waypoint_forward(second_wp, 40)
        CarlaDataProvider._scenario_end_wp = scenario_end_wp

        wps = second_wp.next(self._bicycle_drive_distance)
        if not wps:
            raise ValueError("Couldn't find an end location for the bicycles")
        self._target_locs.append(wps[0].transform.location)

        # Set its initial conditions
        bicycle_2.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(bicycle_2)

    def _create_behavior(self):
        """
        Activate the bicycles and wait for the ego to be close-by before changing the side traffic.
        End condition is based on the ego behind in front of the bicycles, or timeout based.
        """
        root = py_trees.composites.Sequence(name="HazardAtSideLane")
        if self.route_mode:
            total_dist = self._distance + self._obstacle_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        # End condition
        end_condition = py_trees.composites.Sequence(name="End Condition")
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], check_distance=False))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        main_behavior.add_child(end_condition)

        # Bicycle movement. Move them for a set distance, then stop
        offset = self._offset * self._starting_wp.lane_width / 2
        opt_dict = {'offset': offset}
        for actor, target_loc in zip(self.other_actors, self._target_locs):
            bicycle = py_trees.composites.Sequence(name="Bicycle behavior")
            bicycle.add_child(BasicAgentBehavior(actor, target_loc, target_speed=self._bicycle_speed, opt_dict=opt_dict))
            bicycle.add_child(HandBrakeVehicle(actor, 1))  # In case of collisions
            bicycle.add_child(WaitForever())  # Don't make the bicycle stop the parallel behavior
            main_behavior.add_child(bicycle)

        behavior = py_trees.composites.Sequence(name="Side lane behavior")
        behavior.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self.other_actors[0], self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        main_behavior.add_child(behavior)

        root.add_child(main_behavior)
        if self.route_mode:
            root.add_child(SetMaxSpeed(0))

        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

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
        self.remove_all_actors()


class HazardAtSideLaneTwoWays(HazardAtSideLane):
    """
    Variation of the HazardAtSideLane scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_frequency = get_value_parameter(config, 'frequency', float, 200)

        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        Activate the bicycles and wait for the ego to be close-by before changing the opposite traffic.
        End condition is based on the ego behind in front of the bicycles, or timeout based.
        """

        root = py_trees.composites.Sequence(name="HazardAtSideLaneTwoWays")
        if self.route_mode:
            total_dist = self._distance + self._obstacle_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        # End condition
        end_condition = py_trees.composites.Sequence(name="End Condition")
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], check_distance=False))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        main_behavior.add_child(end_condition)

        # Bicycle movement. Move them for a set distance, then stop
        offset = self._offset * self._starting_wp.lane_width / 2
        opt_dict = {'offset': offset}
        for actor, target_loc in zip(self.other_actors, self._target_locs):
            bicycle = py_trees.composites.Sequence(name="Bicycle behavior")
            bicycle.add_child(BasicAgentBehavior(actor, target_loc, target_speed=self._bicycle_speed, opt_dict=opt_dict))
            bicycle.add_child(HandBrakeVehicle(actor, 1))  # In case of collisions
            bicycle.add_child(WaitForever())  # Don't make the bicycle stop the parallel behavior
            main_behavior.add_child(bicycle)

        behavior = py_trees.composites.Sequence(name="Side lane behavior")
        behavior.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self.other_actors[0], self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        behavior.add_child(WaitForever())

        main_behavior.add_child(behavior)

        root.add_child(main_behavior)
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=40))

        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root
