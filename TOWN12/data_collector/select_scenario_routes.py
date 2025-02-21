import glob
import os
import sys
from typing import DefaultDict
import lxml.etree as ET
import argparse
import random
import time

import numpy as np
import tqdm
import carla
import copy

from TOWN12.agents.navigation.global_route_planner import GlobalRoutePlanner

target_scenario = ['ConstructionObstacle', 'ConstructionObstacleTwoWays', 'Accident', 'AccidentTwoWays', 'ParkedObstacle', 'ParkedObstacleTwoWays', 'VehicleOpensDoorTwoWays', 'HazardAtSideLane', 'HazardAtSideLaneTwoWays', 'InvadingTurn']

def weather_string(weather):
    weather_string = f"         <weather\n"
    weather_string += f"            route_percentage=\"0\"\n"
    weather_string += f"            cloudiness=\"{weather.cloudiness}\" "
    weather_string += f"precipitation=\"{weather.precipitation}\" "
    weather_string += f"precipitation_deposits=\"{weather.precipitation_deposits}\" "
    weather_string += f"wetness=\"{weather.wetness}\"\n"
    weather_string += f"            wind_intensity=\"{weather.wind_intensity}\" "
    weather_string += f"sun_azimuth_angle=\"{weather.sun_azimuth_angle}\" "
    weather_string += f"sun_altitude_angle=\"{weather.sun_altitude_angle}\"\n"
    weather_string += f"            fog_density=\"{weather.fog_density}\" "
    weather_string += f"fog_distance=\"{weather.fog_distance}\" "
    weather_string += f"fog_falloff=\"{round(weather.fog_falloff, 2)}\" "
    weather_string += f"scattering_intensity=\"{weather.scattering_intensity}\"\n"
    weather_string += f"            mie_scattering_scale=\"{round(weather.mie_scattering_scale, 2)}\"/>"

    return weather_string

def get_scenario_route_position(route_wps, trigger_location):
    position = 0
    distance = float('inf')
    for i, (wp, _) in enumerate(route_wps):
        route_distance = wp.transform.location.distance(trigger_location)
        if route_distance < distance:
            distance = route_distance
            position = i
    return position

def get_weather(prev_w, next_w, perc):
    def interpolate(prev_w, next_w, perc, name):
        x0 = prev_w[0]
        x1 = next_w[0]
        if x0 == x1:
            raise ValueError("Two weather keypoints have the same route percentage")
        y0 = getattr(prev_w[1], name)
        y1 = getattr(next_w[1], name)
        return y0 + (y1 - y0) * (perc - x0) / (x1 - x0)
    weather = carla.WeatherParameters()
    weather.cloudiness = interpolate(prev_w, next_w, perc, 'cloudiness')
    weather.precipitation = interpolate(prev_w, next_w, perc, 'precipitation')
    weather.precipitation_deposits = interpolate(prev_w, next_w, perc, 'precipitation_deposits')
    weather.wind_intensity = interpolate(prev_w, next_w, perc, 'wind_intensity')
    weather.sun_azimuth_angle = interpolate(prev_w, next_w, perc, 'sun_azimuth_angle')
    weather.sun_altitude_angle = interpolate(prev_w, next_w, perc, 'sun_altitude_angle')
    weather.wetness = interpolate(prev_w, next_w, perc, 'wetness')
    weather.fog_distance = interpolate(prev_w, next_w, perc, 'fog_distance')
    weather.fog_density = interpolate(prev_w, next_w, perc, 'fog_density')
    weather.fog_falloff = interpolate(prev_w, next_w, perc, 'fog_falloff')
    weather.scattering_intensity = interpolate(prev_w, next_w, perc, 'scattering_intensity')
    weather.mie_scattering_scale = interpolate(prev_w, next_w, perc, 'mie_scattering_scale')
    weather.rayleigh_scattering_scale = interpolate(prev_w, next_w, perc, 'rayleigh_scattering_scale')

    return weather

def parse_weather(route):
    """
    Parses all the weather information as a list of [position, carla.WeatherParameters],
    where the position represents a % of the route.
    """
    weathers = []

    weathers_elem = route.find("weathers")
    if weathers_elem is None:
        return [[0, carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)]]

    for weather_elem in weathers_elem.iter('weather'):
        route_percentage = float(weather_elem.attrib['route_percentage'])

        weather = carla.WeatherParameters(sun_altitude_angle=70, cloudiness=50)  # Base weather
        for weather_attrib in weather_elem.attrib:
            if hasattr(weather, weather_attrib):
                setattr(weather, weather_attrib, float(weather_elem.attrib[weather_attrib]))
            elif weather_attrib != 'route_percentage':
                print(f"WARNING: Ignoring '{weather_attrib}', as it isn't a weather parameter")

        weathers.append([route_percentage, weather])

    weathers.sort(key=lambda x: x[0])
    return weathers

def main(args):
    # Get the client
    client = carla.Client('localhost', 3000)
    client.set_timeout(50.0)

    # # Get the rest
    world = client.load_world(args.town)
    tmap = world.get_map()
    grp = GlobalRoutePlanner(tmap, 1)


    def convert_elem_to_location(elem):
        """Convert an ElementTree.Element to a CARLA Location"""
        return carla.Location(float(elem.attrib.get('x')), float(elem.attrib.get('y')), float(elem.attrib.get('z')))
    split_route_list = DefaultDict(list)

    tree = ET.parse(args.origin_file)
    for route in tqdm.tqdm(tree.iter("route")):
        town = route.attrib['town']
        # random choose a weather in between
        weather_list = parse_weather(route)
        perc = random.randint(0,100)
        weather = get_weather(weather_list[0], weather_list[1], perc)
        weather = weather_string(weather)

        route_wps = []
        prev_route_keypoint = None
        # Route data
        for position in route.find('waypoints').iter('position'):
            route_keypoint = convert_elem_to_location(position)
            if prev_route_keypoint:
                route_wps.extend(grp.trace_route(prev_route_keypoint, route_keypoint))
            prev_route_keypoint = route_keypoint

        possible_scenario = []
        for scenario in route.find('scenarios').iter('scenario'):
            type = scenario.attrib.get('type')
            if type not in target_scenario:
                continue
            scenario_name = scenario.attrib.get('name')
            scenario_name = scenario_name.split('_')[:-1] + ['1']
            scenario_name = '_'.join(scenario_name)
            scenario.set('name', scenario_name)
            print(scenario_name)
            possible_scenario.append(scenario)

        for scenario in possible_scenario:
            trigger_location = convert_elem_to_location(scenario.find('trigger_point'))
            scenario_position = get_scenario_route_position(route_wps, trigger_location)
            type = scenario.attrib.get('type')
            split_route_list[type].append([route_wps[scenario_position-50:scenario_position][::10]+route_wps[scenario_position:scenario_position+100][::10], [scenario], town, weather])

    for scenario_type, sub_routes in split_route_list.items():
        print(scenario_type, len(sub_routes))

        root = ET.Element('routes')
        for id, sub_route in enumerate(sub_routes):
            route = ET.SubElement(root, 'route', id='%d'%id, town=sub_route[2])
            weathers = ET.SubElement(route, 'weathers')
            weathers.append(ET.fromstring(sub_route[3]))
            waypoints = ET.SubElement(route, 'waypoints')
            positions = sub_route[0]
            for pos, _ in positions:
                loc = pos.transform.location
                x, y, z = loc.x, loc.y, loc.z
                ET.SubElement(waypoints, 'position', x='%f'%x, y='%f'%y, z='%f'%z)
            scenarios = ET.SubElement(route, 'scenarios')
            for scenario in sub_route[1]:
                scenarios.append(copy.deepcopy(scenario))

        tree = ET.ElementTree(root)
        save_path = os.path.join(args.save_path, scenario_type + '.xml')
        tree.write(save_path, xml_declaration=True, encoding='utf-8', pretty_print=True)



if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_file', type=str, required=False, default="/home/wupenghao/transfuser/leaderboard/data/routes_training.xml", help='xml file path to save the route waypoints')
    parser.add_argument('--save_path', type=str, required=False, default="/home/wupenghao/transfuser/lane_changing_cases", help='xml file path to save the route waypoints')
    parser.add_argument('--town', type=str, default='Town12', help='town for generating routes')
    
    args = parser.parse_args()

    main(args)