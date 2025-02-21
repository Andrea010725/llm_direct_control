from __future__ import print_function

# import ipdb
import py_trees
import carla
import re
import operator
import time

from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from TOWN12.scenario_runner.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, InTriggerDistanceToNextIntersection, ScenarioNPCTriggerFunction, InTriggerDistanceToLocation, DriveDistance
from TOWN12.scenario_runner.srunner.scenarios.basic_scenario import BasicScenario
from TOWN12.scenario_runner.srunner.dynamic_scenarios.functions.record_data import RecordData


from TOWN12.town12_tools.explainable_utils import *

from .functions import *
from time import sleep
from .basic_desc import *
from beta.tool_wenyang.tool_meta.meta_description import _generate_scenario_desc_balance

# 存储上次脚本运行的时间
last_run_time = 0
last_output = None

class BorrowLeftLanePassObstacle(BasicScenario):
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
        self.front_obstacle = None
        self.front_obstacle_wp = None
        self.front_obstacle_loc = None
        self.initialized = False
        self.lane_changed = False
        self.stage = 0
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        self.obstacle_index = -1

        # For: Is success
        self.passed_lane_ids = []


        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#dynamic1': (LEFT_B, [], DEC, [], ''),
            '#dynamic2': (LEFT_B, [], KEEP_S, [], ''),
            '#fix2': (LEFT_B, [], KEEP_S, [], ''),
            '#dynamic3': (LEFT_B, [], KEEP_S, [], ''),
            '#fix3': (KEEP_L, [], ACC, [], ''),
            '#dynamic4': (LEFT_B, [], KEEP_S, [], ''),
            '#fix4': (LEFT_B, [], KEEP_S, [], ''),
            '#dynamic5': (LEFT_B, [], KEEP_S, [], ''),
            '#fix5': (KEEP_L, [], ACC, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("BorrowLeftLanePassObstacle", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

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
        if change_num != 2:
            return False, f'Autopilot变道次数不为2次，实际变道次数为{change_num}次'
        return True, ''

    def uniad_tick_autopilot(self, max_speed):
        ego = self.ego_vehicles[0]
        if self.front_obstacle is None:
            self.front_obstacle = self.other_actors[self.actor_desc.index('obstacle_0')]
            self.front_obstacle_wp = self._map.get_waypoint(self.front_obstacle.get_location())
            self.front_obstacle_loc = self.front_obstacle.get_location()

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(max_speed)
            self._set_ego_autopilot()
            self.initialized = True

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)
        front_obstacle_wp = self._map.get_waypoint(self.front_obstacle.get_location())

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        all_data = {
            'actors': build_actor_data(ego, self.other_actors),
        }

        if self.front_obstacle.is_alive:
            data = Edict(all_data['actors'][self.front_obstacle.id])
            distance = round(self.front_obstacle.get_location().distance(cur_ego_loc), 2)

            # Special: 第一阶段：向左变道
            if self.stage == 0:
                # Desc: 在同一车道
                if cur_wp.lane_id == front_obstacle_wp.lane_id:
                    if distance > 15:
                        self._tf_set_ego_speed(max_speed)
                    elif distance <= 15:
                        if self.lane_changed is False:
                            self._tf_set_ego_offset(-cur_wp.lane_width)
                            print(f'\n>>> force lane change to left')
                            self.lane_changed = True
                        self._tf_set_ego_speed(20)
                    else:
                        print(f'Error: distance = {distance}m')
                        raise NotImplementedError
                else:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        self._tf_set_ego_speed(max_speed)
                    else:
                        self._tf_set_ego_speed(20)

                # Special: 判断是否完成第一阶段
                if self.lane_changed and cur_wp.lane_id != front_obstacle_wp.lane_id and cur_ego_loc.distance(
                        cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 1
                    self.lane_changed = False

            # Special: 第二阶段：借道中
            elif self.stage == 1:
                if cur_wp.lane_id == front_obstacle_wp.lane_id:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        # Desc: 完成借道
                        self._tf_set_ego_speed(max_speed)
                    else:
                        self._tf_set_ego_speed(20)
                else:
                    # Desc: 还在左侧车道
                    if distance > 8 and actor_front_of_ego(data) is False and self.lane_changed is False:
                        self._tf_set_ego_offset(0)
                        self.lane_changed = True
                        print(f'\n>>> force lane change to right')
                    self._tf_set_ego_speed(20)

                # Special: 判断是否完成第二阶段
                if self.lane_changed and cur_wp.lane_id == front_obstacle_wp.lane_id and cur_ego_loc.distance(
                        cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 2
                    self.lane_changed = False

            # Special: 第三阶段：变道回原始车道
            elif self.stage == 2:
                self._tf_set_ego_speed(max_speed)
            else:
                raise NotImplementedError
        else:
            self._tf_set_ego_speed(max_speed)

        return {'cur_wp': cur_wp, 'navi_cmd': 'follow lane'}

    def _get_navi_route(self, cur_wp):
        if cur_wp.lane_id == self.starting_wp.lane_id:
            start_wp = cur_wp
        else:
            start_wp = cur_wp.get_right_lane()
        dest_wp = self.starting_wp.next_until_lane_end(1)[-2].next(200)[0]
        return start_wp, dest_wp

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.front_obstacle is None:
            self.front_obstacle = self.other_actors[self.actor_desc.index('obstacle_0')]
            self.front_obstacle_wp = self._map.get_waypoint(self.front_obstacle.get_location())
            self.front_obstacle_loc = self.front_obstacle.get_location()

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
        front_obstacle_wp = self._map.get_waypoint(self.front_obstacle.get_location())

        RecordData.record_data(ego,cur_wp,self._world,self.other_actors)

        global last_run_time
        global last_output
        #
        current_time = time.time()
        # # 检查是否已经过了5m秒

        if current_time - last_run_time >= 10:
             # 更新上次运行的时间
             last_run_time = current_time
             print("LLM starts!!!")
             print(" current_time - last_run_time = ", current_time - last_run_time)
             RecordData._monitor_loop()
             time.sleep(2)

      #  time.sleep(2)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        info = f'Speed: {int(ego_speed)} km/h '

        if self.front_obstacle.is_alive:
            desc = Edict(explainable_data['actors'][self.front_obstacle.id]['description'])
            data = Edict(explainable_data['actors'][self.front_obstacle.id])
            distance = round(self.front_obstacle.get_location().distance(cur_ego_loc), 2)
            info += f'Distance: {distance}m '

            # Special: 第一阶段：向左变道
            if self.stage == 0:
                # Desc: 在同一车道
                if cur_wp.lane_id == front_obstacle_wp.lane_id:
                    if distance > 30:
                        ego_stage = '#fix1'
                        ego_reason = ''
                        ego_action = 'keep'
                        explainable_desc = f'正常行驶'
                        self._tf_set_ego_speed(20)
                    elif 30 >= distance > 25:
                        ego_stage = '#dynamic1'
                        ego_action = 'decelerate|borrow|lanechange|left|prepare'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            ego_reason = '前方障碍物&右侧有多辆车'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，同时右侧有{right_car_num}辆车，减速并准备向左借道'
                        else:
                            ego_reason = '前方障碍物'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，减速并准备向左借道'
                        self._tf_set_ego_speed(15)
                    elif distance <= 25:
                        if self.lane_changed is False:
                            self._tf_set_ego_offset(-cur_wp.lane_width)
                            print(f'\n>>> force lane change to left')
                            self.lane_changed = True

                        ego_stage = '#dynamic2'
                        ego_action = 'keep|borrow|lanechange|left|stage1'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            ego_reason = '前方障碍物&右侧有多辆车'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，同时右侧有{right_car_num}辆车，向左借道'
                        else:
                            ego_reason = '前方障碍物'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，向左借道'
                    else:
                        print(f'Error: distance = {distance}m')
                        raise NotImplementedError
                else:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        # Desc: 完成变道
                        ego_stage = '#fix2'
                        ego_action = 'keep|borrow|left|stage2'
                        ego_reason = '右侧障碍物'
                        explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，借道中'
                        self._tf_set_ego_speed(20)
                    else:
                        # Desc: 还未完成变道
                        ego_stage = '#dynamic3'
                        ego_action = 'keep|borrow|lanechange|left|stage1'
                        right_car_num = sum([1 for item in self.actor_desc if 'right' in item])
                        if right_car_num > 0:
                            ego_reason = '前方障碍物&右侧有多辆车'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，同时右侧有{right_car_num}辆车，向左借道'
                        else:
                            ego_reason = '前方障碍物'
                            explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，向左借道'
                        self._tf_set_ego_speed(15)

                # Special: 判断是否完成第一阶段
                if self.lane_changed and cur_wp.lane_id != front_obstacle_wp.lane_id and cur_ego_loc.distance(
                        cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 1
                    self.lane_changed = False

            # Special: 第二阶段：借道中
            elif self.stage == 1:
                if cur_wp.lane_id == front_obstacle_wp.lane_id:
                    if cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                        # Desc: 完成借道
                        ego_stage = '#fix3'
                        ego_reason = ''
                        ego_action = 'accelerate'
                        explainable_desc = f'正常行驶'
                        self._tf_set_ego_speed(40)
                    else:
                        # Desc: 还没完成返回原车道
                        ego_stage = '#dynamic4'
                        ego_action = 'keep|borrow|lanechange|right'
                        ego_reason = '右侧障碍物'
                        explainable_desc = f'已通过{desc.type}，向右变道，完成借道中'
                else:
                    # Desc: 还在左侧车道
                    if distance > 8 and actor_front_of_ego(data) is False and self.lane_changed is False:
                        # self._force_ego_lanechange_right()
                        self._tf_set_ego_offset(0)
                        self.lane_changed = True
                        print(f'\n>>> force lane change to right')

                    if self.lane_changed is False:
                        ego_stage = '#fix4'
                        ego_action = 'keep|borrow|left|stage2'
                        ego_reason = '右侧障碍物'
                        explainable_desc = f'{desc.direction}{desc.distance}处有{desc.type}，借道中'
                    else:
                        ego_stage = '#dynamic5'
                        ego_action = 'keep|borrow|lanechange|right'
                        ego_reason = '右侧障碍物'
                        explainable_desc = f'已通过{desc.type}，向右变道，完成借道中'

                # Special: 判断是否完成第二阶段
                if self.lane_changed and cur_wp.lane_id == front_obstacle_wp.lane_id and cur_ego_loc.distance(cur_wp.transform.location) < cur_wp.lane_width / 4.0:
                    self.stage = 2
                    self.lane_changed = False

            # Special: 第三阶段：变道回原始车道
            elif self.stage == 2:
                ego_stage = '#fix5'
                ego_reason = ''
                ego_action = 'accelerate'
                explainable_desc = f'正常行驶'
                self._tf_set_ego_speed(40)
            else:
                raise NotImplementedError
        else:
            self._tf_set_ego_speed(40)
            ego_stage = '#fix6'
            ego_reason = ''
            ego_action = 'keep'
            explainable_desc = f'正常行驶'

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = 'follow lane'
        explainable_data['info'] = info

        meta_input = {'sce_name': self.name, 'chuyang_inte': explainable_data}
        reason = _generate_scenario_desc_balance(meta_input)['description_demo_eng']

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{env_description}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        # Depend: 场景保证左侧可以借道
        lane_info = get_lane_info(self.starting_wp)

        # For: 生成NPC
        nearby_spawn_points = get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 5))

        # Desc: 前方生成障碍物（如果是车辆的话，一定是侧翻的车辆）
        # Special: 这里和JiangWei讨论后，决定借道的场景前方一定是静态无法移动的物体
        ob_wp = move_waypoint_forward(self.starting_wp, random.randint(30, 40))

        possible_obstacles = ['box', 'cone']
        # possible_obstacles = ['barrier']
        obstacle_type = choice(possible_obstacles)
        if obstacle_type == 'box':
            obstacles = box_obstacle_one_lane_scenario(ob_wp)
        elif obstacle_type == 'cone':
            obstacles = cone_obstacle_one_lane_scenario(ob_wp)
        elif obstacle_type == 'barrier':
            obstacles = barrier_obstacle_one_lane_scenario(ob_wp)
        else:
            raise NotImplementedError

        for ob_index, (bp_name, bp_transform) in enumerate(obstacles):
            self.actor_desc.append(f'obstacle_{ob_index}')
            obstacle = CarlaDataProvider.request_new_actor(bp_name, bp_transform)
            self.other_actors.append(obstacle)
            obstacle.set_simulate_physics(enabled=True)
            if self.obstacle_index == -1:
                self.obstacle_index = len(self.other_actors) - 1

        # For: 右侧如果存在车道，80%的概率生成车流
        if lane_info.r2l > 1 and random.random() < 1.0:
            right_vehicles = right_lane_vehicle_flow_scenario(self.starting_wp)
            for v_index, (v_bp, v_transform) in enumerate(right_vehicles):
                right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
                self.other_actors.append(right_actor)
                self.actor_desc.append(f'right_{v_index}')

        for v_index, (v_bp, v_transform) in enumerate(nearby_spawn_points):
            npc_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform)
            if npc_actor is not None:
                self.other_actors.append(npc_actor)
                self.actor_desc.append(f'npc_{v_index}')

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()

    def start_npc(self):
        for index, (a_desc, actor) in enumerate(zip(self.actor_desc, self.other_actors)):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                if 'npc' in a_desc:
                    self.traffic_manager.set_desired_speed(actor, random.randint(15, 35))
                else:
                    self.traffic_manager.set_desired_speed(actor, 15)
                self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="BorrowLeftLanePassObstacle", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sub_root = py_trees.composites.Sequence(name="BorrowLeftLanePassObstacle_sub_root")
        start_condition = InTriggerDistanceToLocation(self.ego_vehicles[0], self.starting_wp.transform.location, self.starting_wp.lane_width)
        sub_root.add_child(start_condition)
        npc_function = ScenarioNPCTriggerFunction(self)
        sub_root.add_child(npc_function)
        end_condition = py_trees.composites.Parallel(name="BorrowLeftLanePassObstacle", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="BorrowLeftLanePassObstacle_c1")
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.obstacle_index], self.ego_vehicles[0], 10))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.obstacle_index], self.ego_vehicles[0], 40, comparison_operator=operator.gt))
        end_condition.add_child(c1)
        end_condition.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 8))
        sub_root.add_child(end_condition)
        root.add_child(sub_root)
        root.add_child(DriveDistance(self.ego_vehicles[0], 100))
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


