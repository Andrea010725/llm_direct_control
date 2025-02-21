#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario with low visibility, the ego performs a turn only to find out that the end is blocked by another vehicle.
"""

from __future__ import print_function

import math
from TOWN12.town12_tools.explainable_utils import *
import carla
import py_trees
from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy, Idle, ScenarioTimeout)
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle

from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.tools.background_manager import HandleJunctionScenario

from TOWN12.scenario_runner.srunner.tools.scenario_helper import generate_target_waypoint_in_route


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location


class BlockedIntersection(BasicScenario):
    """
    This class holds everything required for a scenario in which,
    the ego performs a turn only to find out that the end is blocked by another vehicle.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)

        self._blocker_distance = 7
        self._trigger_distance = 12
        self._stop_time = 10

        self._scenario_timeout = 240
        self.switched = 0

        super().__init__("BlockedIntersection",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        interact_actor = self.other_actors[0]
        # Template: print(f'Distance: {distance}  Speed: {ego_speed} km/h  Desc: Enter Junction -> Low Speed')

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors),
            # 'nick_name': {
            #     interact_actor.id: 'block_car',
            #     ego.id: 'ego'
            # }
        }

        ego_waypoint = get_actor_waypoint(self.world, ego)
        self._tf_set_ego_speed(20)

        if interact_actor.is_alive:
            actor_data = explainable_data['actors'][interact_actor.id]
            desc = actor_data['description']
            print(f'\rDistance: {round(actor_data["distance"], 2)}  Speed: {ego_speed} km/h ', end='')

            if actor_data['cos_value'] > 0:
                if ego_speed > 0.1:
                    ego_stage = '#dynamic1'
                    ego_action = 'turn|decelerate|junction'
                    explainable_desc = f'路口{desc["direction"]}{desc["distance"]}有一辆{desc["color"]}{desc["type"]}{desc["speed"]}，减速'
                else:
                    if actor_data['distance'] > 5:
                        if self.switched <= 1:
                            ipdb.set_trace(context=10)
                            # self._tf_switch_ego_autopilot()
                            self.switched += 1
                            # print(f'\nswitch ego mode: {self.ego_is_autopilot}\n')
                    ego_stage = '#dynamic2'
                    ego_action = 'turn|stop|junction'
                    explainable_desc = f'路口{desc["direction"]}{desc["distance"]}有一辆{desc["color"]}{desc["type"]}{desc["speed"]}，等待前车通过路口'
            else:
                ego_stage = '#error'
                ego_action = 'unknown'
                explainable_desc = f'正常行驶'
        else:
            print(f'\rDistance: NaN  Speed: {ego_speed} km/h ', end='')
            if ego_waypoint.is_intersection or ego_waypoint.is_junction:
                ego_stage = '#fix2'
                ego_action = 'turn|'
                explainable_desc = f'进入路口，减速'
            else:
                ego_stage = '#fix3'
                ego_action = 'hold|normalspeed'
                explainable_desc = f'正常行驶'
        explainable_data['explainable_desc'] = explainable_desc
        explainable_data['ego_stage'] = ego_stage
        explainable_data['scenario'] = 'OppositeVehicleRunningRedLight'
        explainable_data['ego_action'] = ego_action
        print(f' {ego_stage}: {explainable_desc}          ', end='')
        return explainable_data

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = generate_target_waypoint_in_route(self._reference_waypoint, config.route)
        waypoint = waypoint.next(self._blocker_distance)[0]

        # Spawn the blocker vehicle
        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*", waypoint.transform,
            attribute_filter={'base_type': 'car', 'has_lights': True}
        )
        if actor is None:
            raise Exception("Couldn't spawn the blocker vehicle")
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        Just wait for a while after the ego closes in on the blocker, then remove it.
        """
        sequence = py_trees.composites.Sequence(name="BlockedIntersection")

        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[],
                remove_exits=[],
                stop_entries=True,
                extend_road_exit=0
            ))
        # Ego go behind the blocker
        main_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        behavior = py_trees.composites.Sequence(name="Approach and Wait")
        behavior.add_child(InTriggerDistanceToVehicle(
            self.other_actors[-1], self.ego_vehicles[0], self._trigger_distance))
        behavior.add_child(Idle(self._stop_time))
        main_behavior.add_child(behavior)

        sequence.add_child(main_behavior)
        sequence.add_child(ActorDestroy(self.other_actors[-1]))

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
        self.remove_all_actors()
