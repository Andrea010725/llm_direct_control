import glob
import os
import sys
import lxml.etree as ET
import argparse
import random
import time

import numpy as np
import tqdm
import carla
import copy
# from TOWN12.agents.navigation.global_route_planner import GlobalRoutePlanner

remaining_scenarios = ['DynamicObjectCrossing', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian',
'MergerIntoSlowTraffic', 'MergerIntoSlowTrafficV2', 'CrossingBicycleFlow', 'HighwayCutIn', 'StaticCutIn', 'ControlLoss', 'HardBreakRoute']

WEATHER_LIST = {'ClearNoon':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudyNoon':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetNoon':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudyNoon':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 180.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainNoon':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 315.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainyNoon':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.40,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainNoon':{'cloudiness': 90.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1.0,
                             'sun_azimuth_angle': 90.0,
                             'sun_altitude_angle': 75.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'ClearSunset':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudySunset':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetSunset':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudySunset':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainSunset':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.40,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainSunset':{'cloudiness': 80.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1.0,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainSunset':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': 15.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'ClearNight':{'cloudiness': 15.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 0.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'CloudyNight':{'cloudiness': 80.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 0.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 45.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetNight':{'cloudiness': 20.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'WetCloudyNight':{'cloudiness': 90.0,
                             'precipitation': 0.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'SoftRainNight':{'cloudiness': 90.0,
                             'precipitation': 15.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity': 0.35,
                             'sun_azimuth_angle': 270.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'MidRainyNight':{'cloudiness': 80.0,
                             'precipitation': 30.0,
                             'precipitation_deposits': 50.0,
                             'wind_intensity':0.4,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                'HardRainNight':{'cloudiness': 90.0,
                             'precipitation': 60.0,
                             'precipitation_deposits': 100.0,
                             'wind_intensity':1,
                             'sun_azimuth_angle': 225.0,
                             'sun_altitude_angle': -80.0,
                             'fog_density': 0.0, 
                             'fog_distance': 0.0,
                             'fog_falloff': 0.0,
                             'wetness': 0.0,
                            },
                }





# def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0):
#     """
#     Given some raw keypoints interpolate a full dense trajectory to be used by the user.
#     Args:
#         world: an reference to the CARLA world so we can use the planner
#         waypoints_trajectory: the current coarse trajectory
#         hop_resolution: is the resolution, how dense is the provided trajectory going to be made
#     Return: 
#         route: full interpolated route both in GPS coordinates and also in its original form.
#     """

#     grp = GlobalRoutePlanner(world_map, hop_resolution)
#     # Obtain route plan
#     route = []
#     is_junction = False
#     distance = 0

#     for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

#         waypoint = waypoints_trajectory[i]
#         waypoint_next = waypoints_trajectory[i + 1]
#         interpolated_trace = grp.trace_route(waypoint, waypoint_next)
#         for i, wp_tuple in enumerate(interpolated_trace):
#             route.append(wp_tuple[0].transform)
#             if i > 0:
#                 distance += wp_tuple[0].transform.location.distance(interpolated_trace[i-1][0].transform.location)
#             # print (wp_tuple[0].transform.location, wp_tuple[1])

#     return distance


def main():
    # client = carla.Client('localhost', 2000)
    # client.set_timeout(200.0)
    # world = client.load_world(args.town)
    # world_map = world.get_map()
    # print ('loaded world')

    thre_dist = 300

    split_route_list = []

    tree = ET.parse(args.origin_file)
    for route in tqdm.tqdm(tree.iter("route")):
        town = route.attrib['town']
        weathers = route.find('weathers')

        # The list of carla.Location that serve as keypoints on this route
        positions = []
        for position in route.find('waypoints').iter('position'):
            positions.append(carla.Location(x=float(position.attrib['x']),
                                            y=float(position.attrib['y']),
                                            z=float(position.attrib['z'])))

        
        possible_scenario = []
        for scenario in route.find('scenarios').iter('scenario'):
            type = scenario.attrib.get('type')
            if type not in remaining_scenarios:
                continue
            possible_scenario.append(scenario)

        start_iter = 0
        while start_iter < len(positions)-3:
            split_route_list.append([positions[start_iter:start_iter+3], town, weathers, possible_scenario])
            start_iter += 2
        split_route_list.append([positions[start_iter:], town, weathers, possible_scenario])

    total_number = len(split_route_list)
    print(total_number)

    number_of_file = int(np.ceil(total_number/args.route_num))

    for i in range(number_of_file):
        sub_routes = split_route_list[i*args.route_num: (i+1)*args.route_num]
        root = ET.Element('routes')
        for id, sub_route in enumerate(sub_routes):
            route = ET.SubElement(root, 'route', id='%d'%id, town=sub_route[1])
            route.append(copy.deepcopy(sub_route[2]))
            waypoints = ET.SubElement(route, 'waypoints')
            positions = sub_route[0]
            for pos in positions:
                x, y, z = pos.x, pos.y, pos.z
                ET.SubElement(waypoints, 'position', x='%f'%x, y='%f'%y, z='%f'%z)
            scenarios = ET.SubElement(route, 'scenarios')
            for scenario in sub_route[-1]:
                scenarios.append(copy.deepcopy(scenario))

        tree = ET.ElementTree(root)
        save_path = args.save_file + '_' + str(i) +'.xml'
        tree.write(save_path, xml_declaration=True, encoding='utf-8', pretty_print=True)

if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_file', type=str, required=False, default="/home/wupenghao/transfuser/lb2/leaderboard/data/routes_training.xml", help='xml file path to save the route waypoints')
    parser.add_argument('--save_file', type=str, required=False, default="/home/wupenghao/transfuser/lb2/leaderboard/data/routes_training_split", help='xml file path to save the route waypoints')
    parser.add_argument('--town', type=str, default='Town12', help='town for generating routes')
    parser.add_argument('--route_num', type=int, default=300, help='number of routes')
    
    args = parser.parse_args()

    main()
