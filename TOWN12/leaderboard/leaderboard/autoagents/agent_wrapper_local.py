#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os

import carla
from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from TOWN12.leaderboard.leaderboard.autoagents.autonomous_agent import Track

from TOWN12.leaderboard.leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
# Use this line instead to run WOR
#from leaderboard1.leaderboard.envs.sensor_interface import (CallBack, StitchCameraReader, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid)
from TOWN12.leaderboard.leaderboard.autoagents.autonomous_agent import Track
from TOWN12.leaderboard.leaderboard.autoagents.ros_base_agent import ROSBaseAgent

# DATAGEN = int(os.environ.get('DATAGEN'))
MAX_ALLOWED_RADIUS_SENSOR = 10.0

SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1,
    'sensor.stitch_camera.rgb': 1,
    'sensor.camera.depth': 4, # for data generation
    'sensor.camera.semantic_segmentation': 4 # for data generation
}

ALLOWED_SENSORS = SENSORS_LIMITS.keys()

class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class AgentWrapperFactory(object):

    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)

def validate_sensor_configuration(sensors, agent_track, selected_track):
    """
    Ensure that the sensor configuration is valid, in case the challenge mode is used
    Returns true on valid configuration, false otherwise
    """
     
    # if selected_track != agent_track:
    #     raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))
    return
    sensor_count = {}
    sensor_ids = []

    for sensor in sensors:

        # Check if the is has been already used
        sensor_id = sensor['id']
        if sensor_id in sensor_ids:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
        else:
            sensor_ids.append(sensor_id)

        # Check if the sensor is valid
        if agent_track == Track.SENSORS:
            if sensor['type'].startswith('sensor.opendrive_map'):
                raise SensorConfigurationInvalid("Illegal sensor used for Track [{}]!".format(agent_track))

        # Check the sensors validity
        if sensor['type'] not in ALLOWED_SENSORS:
            raise SensorConfigurationInvalid("Illegal sensor used. {} are not allowed!".format(sensor['type']))

        # Check the extrinsics of the sensor
        if 'x' in sensor and 'y' in sensor and 'z' in sensor:
            if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                raise SensorConfigurationInvalid(
                    "Illegal sensor extrinsics used for Track [{}]!".format(agent_track))

        # Check the amount of sensors
        if sensor['type'] in sensor_count:
            sensor_count[sensor['type']] += 1
        else:
            sensor_count[sensor['type']] = 1

    for sensor_type, max_instances_allowed in SENSORS_LIMITS.items():
        if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
            raise SensorConfigurationInvalid(
                "Too many {} used! "
                "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                              max_instances_allowed,
                                                                              sensor_count[sensor_type]))

class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """
    _agent = None
    _sensors_list = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self):
        """
        Pass the call directly to the agent
        """

        return self._agent()

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_.startswith('sensor.camera'):
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['lens_circle_multiplier'] = str(3.0)
            attributes['lens_circle_falloff'] = str(3.0)
            if sensor_spec['type'].startswith('sensor.camera.rgb'):
                attributes['chromatic_aberration_intensity'] = str(0.5)
                attributes['chromatic_aberration_offset'] = str(0)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        world = CarlaDataProvider.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        for _ in range(10):
            world.tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        # Tick once to destroy the sensors
        CarlaDataProvider.get_world().tick()