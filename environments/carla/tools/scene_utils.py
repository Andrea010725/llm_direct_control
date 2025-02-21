import ipdb
from functions.world_utils import WorldManager
from functions.spectator_utils import SpectatorManager
from functions.actor_utils import ActorManager
from functions.scenario_utils import ScenarioManager
from functions.visual_utils import VisualManager
from functions.route_utils import RouteManager
from pygame.draw_py import clip_line
import carla
from time import sleep
from addict import Dict

ROUTE_INDEX = 1
TEST_WP_INDEX = 0

class SceneManager(object):
    def __init__(self, selected_task, world, client, vehicle):
        self.selected_task = selected_task
        self.world = world
        self.client = client
        self.vehicle = vehicle

    def create_scene(self):
        xml_file = f"all_cases_updated/{self.selected_task}.xml"
        route_manager = RouteManager(self.world, xml_file)

        spectator_manager = SpectatorManager(self.world)
        visual_manager = VisualManager(self.world.debug, spectator_manager)
        traffic_manager = self.client.get_trafficmanager(8000)
        actor_manager = ActorManager(self.world, traffic_manager)

        wp_route = route_manager.get_route(0, visual_manager=visual_manager)
        route_info = route_manager.routes[ROUTE_INDEX]

        function_name = f"reproduce_supply_{self.selected_task.lower()}"  # 动态生成函数名
        # 检查函数是否存在于当前作用域
        if function_name in globals():
            # 动态调用函数
            globals()[function_name](wp_route, self.client, self.world, actor_manager, spectator_manager, traffic_manager,
                                     route_info, self.vehicle)
        else:
            raise ValueError(f"Function '{function_name}' is not defined.")

def reproduce_supply_junctionleftlanechangeone(wp_route, client, world, actor_manager, spectator_manager,
                                               traffic_manager, route_info, vehicle):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.junctionleftlanechangeone_condition import JunctionLeftLaneChangeOneCondition
    scenario_condition = JunctionLeftLaneChangeOneCondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)

    path = wp_route  # 稀疏化wp
    ego = vehicle
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('JunctionLeftLaneChangeOne_dynamic', client, world, actor_manager,
                                       spectator_manager, traffic_manager, scenario_config)
    scenario_manager.debug(wp_route[0][0].transform, path, transform_route=transform_route)

def reproduce_supply_borrowrightlanepassobstacle(wp_route, client, world, actor_manager, spectator_manager,
                                                 traffic_manager, route_info):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.borrowrightlanepassobstacle_condition import \
        BorrowRightLanePassObstacleCondition
    scenario_condition = BorrowRightLanePassObstacleCondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)

    path = wp_route

    ego_spawn_wp = wp_route[0][0]

    ego = actor_manager.spawn_vehicle(ego_spawn_wp, hero=True)
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('BorrowRightLanePassObstacle_dynamic', client, world, actor_manager,
                                       spectator_manager, traffic_manager, scenario_config)
    scenario_manager.debug(wp_route[0][0].transform, path)

def reproduce_supply_overtakevehiclefromleft(wp_route, client, world, actor_manager, spectator_manager,
                                             traffic_manager, route_info):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.overtakevehiclefromleft_condition import \
        OvertakeVehicleFromLeftCondition
    scenario_condition = OvertakeVehicleFromLeftCondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)
    path = wp_route

    ego_spawn_wp = wp_route[0][0]

    ego = actor_manager.spawn_vehicle(ego_spawn_wp, hero=True)
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('OvertakeVehicleFromLeft_dynamic', client, world, actor_manager,
                                       spectator_manager, traffic_manager, scenario_config)
    scenario_manager.debug(wp_route[0][0].transform, path)

def reproduce_supply_giveawayemergencyright(wp_route, client, world, actor_manager, spectator_manager,
                                            traffic_manager, route_info):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.giveawayemergencyright_condition import \
        GiveAwayEmergencyRightCondition
    scenario_condition = GiveAwayEmergencyRightCondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)

    path = wp_route

    ego_spawn_wp = wp_route[0][0]

    ego = actor_manager.spawn_vehicle(ego_spawn_wp, hero=True)
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('GiveAwayEmergencyRight_dynamic', client, world, actor_manager,
                                       spectator_manager, traffic_manager, scenario_config)
    scenario_manager.debug(wp_route[0][0].transform, path)

def reproduce_supply_ghosta(wp_route, client, world, actor_manager, spectator_manager, traffic_manager, route_info):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.ghosta_condition import GhostACondition
    scenario_condition = GhostACondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)

    path = wp_route

    ego_spawn_wp = wp_route[0][0]

    ego = actor_manager.spawn_vehicle(ego_spawn_wp, hero=True)
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('GhostA_dynamic', client, world, actor_manager, spectator_manager,
                                       traffic_manager, scenario_config)
    scenario_manager.debug(wp_route[0][0].transform, path)

def reproduce_supply_obstacleahead(wp_route, client, world, actor_manager, spectator_manager, traffic_manager,
                                   route_info, vehicle):
    trigger_point = route_info['scenario']['trigger_point']
    case_trigger_wp = world.get_map().get_waypoint(
        location=carla.Location(
            x=float(trigger_point['x']),
            y=float(trigger_point['y']),
            z=float(trigger_point['z']),
        )
    )

    from leaderboard.scenarios.dynamic_scenarios.scenario_condition_dynamic.obstacleahead_condition import \
        ObstacleAheadCondition
    scenario_condition = ObstacleAheadCondition(wp_route[TEST_WP_INDEX][0])  # 输入目前自车所在的wp idx
    wp_route, next_start_index, scenario_config = scenario_condition.apply_scenario(wp_route,
                                                                                    trigger_wp=case_trigger_wp)

    path = wp_route

    ego = vehicle
    sleep(0.5)
    actor_manager.set_ego(ego)

    transform_route = [(item[0].transform, item[1]) for item in wp_route]

    # 根据Condition随机返回的front distance更新config
    scenario_config = Dict({
        'start_distance': 0,
        'random': True,
        'wp_route': wp_route,
        'route_var_name': None,
    })
    scenario_manager = ScenarioManager('ObstacleAhead_dynamic', client, world, actor_manager, spectator_manager,
                                       traffic_manager, scenario_config)
    ipdb.set_trace()
    scenario_manager.debug(wp_route[0][0].transform, path, vehicle)



