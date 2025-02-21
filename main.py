# 启carla  生成自车
# 动作对应
import sys
sys.path.append(r'D:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-cp37-cp37m-win_amd64.whl')
import carla
import time
import math
import xml.etree.ElementTree as ET
import random
import ipdb
import json

# 修改成自己的carla路径
sys.path.append(r'D:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController, PIDLongitudinalController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector, is_within_distance, get_speed
from basic_senerio import BasicScenario
from TOWN12.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from automatic_gpt import get_gpt_result
from environments.carla import env_utils

class decision_define():
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.traffic_manager = self.client.get_trafficmanager(8000)

    def path_decision_define(self,task,ego_vehicle):
        """
        Path decision definitions:
        'LEFT_LANE_CHANGE' - Change to left lane
        'RIGHT_LANE_CHANGE' - Change to right lane
        'LEFT_LANE_OVERTAKE' - Temporarily use left lane
        'RIGHT_LANE_OVERTAKE' - Temporarily use right lane
        'LANE_KEEP' - Maintain current lane
        """
        control_dict = {
            'LEFT_LANE_CHANGE': lambda vehicle: (
                self.traffic_manager.force_lane_change(vehicle, False),
                True),
            'RIGHT_LANE_CHANGE': lambda vehicle:(
                self.traffic_manager.force_lane_change(vehicle, True),
                True),
            'LEFT_LANE_OVERTAKE': lambda vehicle: (
               self.traffic_manager.force_lane_change(vehicle, False),
               True),
            'RIGHT_LANE_OVERTAKE': lambda vehicle: (
                self.traffic_manager.force_lane_change(vehicle, True),
                True
            ),
            'LANE_KEEP': lambda vehicle: (
                vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0)),
                True
            ),
        }
        return control_dict.get(task, None)

    def speed_decision_define(self, task):
        """
        Speed decision definitions:
        'ACCELERATE' - Increase speed
        'DECELERATE' - Decrease speed
        'MAINTAIN' - Maintain current speed
        'STOP' - Stop vehicle
        """
        control_dict = {
            'ACCELERATE': lambda vehicle: (
                vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0)),
                True
            ),
            'DECELERATE': lambda vehicle: (
                vehicle.apply_control(carla.VehicleControl(throttle=0.05, brake=0.3)),
                True
            ),
            'MAINTAIN': lambda vehicle: (
                vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0)),
                True
            ),
            'STOP': lambda vehicle: (
                vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0)),
                True
            )
        }
        return control_dict.get(task, None)



class CarlaWorld(decision_define):
    def __init__(self):
        super().__init__()
        self.world = self.client.load_world('Town05')    #  暂定都是Town05
        # self.world = self.client.get_world()
        self.map = self.world.get_map()
        # 开启同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)  # 需要应用设置
        

        # # 初始化traffic manager
        # self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        

    def generate_tasks(self):
        # 定义任务列表
        Tasks_list = [
            'PrecedingVehicleStationary',
            'FrontBrake',
            'JunctionPedestrianCross',
            'GiveAwayEmergencyRight',
            'GhostA',
            'ObstacleAhead'
        ]

        # 随机选择一个任务
        # selected_task = random.choice(Tasks_list)  #不debug的时候打开
        selected_task = 'FrontBrake'    #only for debug
        print(f'Selected task: {selected_task}')
        function_name = f"reproduce_supply_{selected_task.lower()}"  # 动态生成函数名

        # 起始点
        #xml_file = f"all_cases_updated/{selected_task}.xml"
        xml_file = f"debug_cases/{selected_task}.xml"
        trigger_point = self.get_random_waypoint(xml_file)
        print('trigger point', trigger_point)
        return trigger_point, selected_task
    
    def get_random_waypoint(self, xml_file):
        """
        从 XML 文件中随机选择一个 waypoint，并返回其 x, y, z 坐标。
        """
        # 解析 XML 文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 提取所有 waypoints
        waypoints = []
        for waypoint_elem in root.findall('./route/waypoints/position'):
            waypoints.append({
                'x': float(waypoint_elem.get('x')),
                'y': float(waypoint_elem.get('y')),
                'z': float(waypoint_elem.get('z')),
                'pitch': 0,
                'yaw': float(waypoint_elem.get('yaw')),
                'roll': 0
            })

        # 随机选择一个 waypoint
        if waypoints:
            random_waypoint = random.choice(waypoints)
            return random_waypoint
        else:
            raise ValueError("No waypoints found in the XML file.")
        
    
    def spawn_actors_debug(self,  origin_location: carla.Location, selected_task: str, hybrid=True, safe=True):
        """Instantiate vehicles and pedestrians in the current world"""
        print('selected_task',selected_task)
        origin = carla.Location(x=origin_location.x,y=origin_location.y,z=origin_location.z)
        ego_wp =self.world.get_map().get_waypoint(location=origin, project_to_road=True, lane_type=carla.LaneType.Driving)
      #  ego_wp = self.world.get_map().get_waypoint(origin)
        task = selected_task
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

      
        """
            'PrecedingVehicleStationary',
            'FrontBrake',
            'JunctionPedestrianCross',
            'GiveAwayEmergencyRight',
            'GhostA',
            'ObstacleAhead'
        """
        if task == 'PrecedingVehicleStationary':
            self.special_scene = env_utils.PrecedingVehicleStationary(self.client, self.world, ego_wp)

        if task == 'FrontBrake':
            self.special_scene = env_utils.FrontBrake(self.client,self.world, ego_wp)

        if task == 'JunctionPedestrianCross':
            self.special_scene = env_utils.JunctionPedestrianCross(self.world, ego_wp)

        if task == 'GiveAwayEmergencyRight':
            self.special_scene = env_utils.GiveAwayEmergencyRight(self.world, ego_wp)

        if task == 'GhostA':
            self.special_scene = env_utils.GhostA(self.world, ego_wp)

        if task == 'ObstacleAhead':
            self.special_scene = env_utils.ObstacleAhead(self.client, self.world, ego_wp)
            #print(f"!!!!!!!!!{len(self.world.get_actors().filter('vehicle.*'))}")

    
    def spawn_ego_vehicle(self):
        self.origin_point, self.selected_task = self.generate_tasks()
        
        x = self.origin_point['x']
        y = self.origin_point['y']
        z = self.origin_point['z']
        yaw = self.origin_point['yaw']
        
        self.origin_location = carla.Location(x=x, y=y, z=z)
        print("!!!!!origin_location!!!!!!", self.origin_location)
        self.origin = carla.Transform(location=carla.Location(x=x, y=y, z=z+0.5), rotation=carla.Rotation(pitch=0, yaw=yaw, roll=0))

        vehicle_filter='vehicle.tesla.model3'
        print("self.origin = ",self.origin)
        blueprint = env_utils.random_blueprint(self.world, actor_filter=vehicle_filter)
        # 设置车辆颜色为红色
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '255,0,0')
        ego_vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, self.origin)
        
        # 添加等待时间确保车辆完全生成
        self.world.tick()
        time.sleep(0.5)  # 给予一些时间让车辆完全生成
        
        # 验证车辆位置
        if ego_vehicle:
            actual_location = ego_vehicle.get_location()
            print(f"Actual ego vehicle location: x={actual_location.x}, y={actual_location.y}, z={actual_location.z}")
            
            # 如果位置为0，说明生成可能有问题
            if abs(actual_location.x) < 0.1 and abs(actual_location.y) < 0.1:
                print("Warning: Vehicle position abnormal, spawn may have failed!")
        
        return ego_vehicle, self.origin_location

    def generate_spectator(self):
        self.spectator = self.world.get_spectator()
        x=self.origin_location.x 
        y=self.origin_location.y
       
        
        #self.spectator.set_transform(carla.Transform(location=carla.Location(x=x-5, y=y+20, z=z), rotation=carla.Rotation(yaw= -90,pitch= 0)))   # check一下角度
        #self.spectator.set_transform(carla.Transform(location=carla.Location(x=x, y=y, z=50), rotation=carla.Rotation(pitch= -90)))   # ObstacleAhead
        self.spectator.set_transform(carla.Transform(location=carla.Location(x=x, y=y, z=60), rotation=carla.Rotation(yaw= -90,pitch=-90)))   # FrontBrake

        return self.spectator
    
    def check_previous_close_participants(self,ego_vehicle):
        """
        检查自车周围的所有车辆、静态物体和行人信息
        """
        surrounding_info = []
        surrounding_actors = []
        N = 0

        if ego_vehicle is None:
            print("Cannot find ego vehicle!")
            return surrounding_info, surrounding_actors

        # 获取自车位置
        hero_location = ego_vehicle.get_location()
        print("Ego vehicle location:", hero_location)

        # 遍历所有 Actor
        for actor in self.world.get_actors():
            # 如果是自车，跳过
            if actor.id == ego_vehicle.id or actor.type_id == 'spectator' or actor.type_id == 'traffic':
                continue

            # 记录周围 Actor
            surrounding_actors.append(actor)

            # 获取 Actor 的位置和速度（如果有）
            actor_location = actor.get_location()
            actor_velocity = actor.get_velocity() if actor.type_id.startswith('vehicle') or actor.type_id.startswith(
                'walker') else None

            # 计算与自车的距离
            distance = math.sqrt(
                (hero_location.x - actor_location.x) ** 2 +
                (hero_location.y - actor_location.y) ** 2 +
                (hero_location.z - actor_location.z) ** 2
            )

            # 仅记录距离在 50 米以内的 Actor
            if distance <= 50:
                actor_info = {
                    'type': actor.type_id,  # Actor 类型（如 vehicle.* 或 walker.*）
                    'location': {
                        'x': actor_location.x,
                        'y': actor_location.y,
                        'z': actor_location.z
                    },
                    'distance': distance
                }

                # 如果 Actor 有速度信息，添加速度信息
                if actor_velocity is not None:
                    actor_info['velocity'] = {
                        'x': actor_velocity.x,
                        'y': actor_velocity.y,
                        'z': actor_velocity.z
                    }

                # 判断是否在自车前方
                if self.is_in_front_of_hero_vehicle(ego_vehicle,actor):
                    actor_info['is_in_front'] = True
                else:
                    actor_info['is_in_front'] = False

                surrounding_info.append(actor_info)

                # Debug 信息
                if N == 0:
                   # print(f"当前场景中共有 {len(world.get_actors())} 个 Actor！")
                    N += 1

        return surrounding_info, surrounding_actors
    
    def is_in_front_of_hero_vehicle(self, ego_vehicle,other_vehicle):
        hero_location = ego_vehicle.get_location()
        hero_rotation = ego_vehicle.get_transform().rotation
        other_location = other_vehicle.get_location()
        relative_position = carla.Vector3D(
            other_location.x - hero_location.x,
            other_location.y - hero_location.y,
            other_location.z - hero_location.z
        )
        hero_front_vector = hero_rotation.get_forward_vector()
        k = 0.8
        l = 10
        if other_location.x <= hero_location.x + l * math.cos(hero_rotation.yaw):
            if other_location.y <= hero_location.y + l * math.sin(hero_rotation.yaw):
                return True
        return False


    
    def ego_control(self,ego_vehicle):
        """
        根据GPT输出的决策执行相应的车辆控制
        """
        try:
            # 检查文件是否存在,如果存在则读取已有数据,不存在则创建空列表
            all_positions = []
            try:
                with open('scene_data/all_ego_positions.json', 'r', encoding='utf-8') as f:
                    all_positions = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_positions = []
            
            while True:  # 持续运行
                # 获取并记录当前时刻的位置信息
                current_location = ego_vehicle.get_location()
                current_velocity = ego_vehicle.get_velocity()
                timestamp = time.time()
                
                position_data = {
                    "timestamp": timestamp,
                    "position": {
                        "x": current_location.x,
                        "y": current_location.y,
                        "z": current_location.z
                    },
                    "velocity": {
                        "x": current_velocity.x,
                        "y": current_velocity.y,
                        "z": current_velocity.z
                    }
                }
                # 追加新的位置数据
                all_positions.append(position_data)
                
                # 保存更新后的完整数据到文件
                with open('scene_data/all_ego_positions.json', 'w', encoding='utf-8') as f:
                    json.dump(all_positions, f, indent=4, ensure_ascii=False)
                
                # 原有的控制逻辑
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
                print("Waiting for GPT decision...")
                
                # 获取当前的信息 存在json 给LLM 
                # 获取ego 位置
                vehicle_velocity = ego_vehicle.get_velocity()
                ego_info = {
                    "vehicle_location":{
                        "x": ego_vehicle.get_location().x,
                        "y": ego_vehicle.get_location().y,
                        "z": 0.3 # ego_vehicle.get_location().z
                    },
                    "vehicle_velocity":{
                        "x": vehicle_velocity.x,
                        "y": vehicle_velocity.y,
                        "z": vehicle_velocity.z
                    }
                }
                print("ego_info",ego_info)
                ego_info_path = 'D:\\Github仓库\\llm_direct_control\\scene_data\\ego_position.json'
                with open(ego_info_path, 'w') as f:
                    json.dump(ego_info, f, indent=4)
                    #print(f"ego_info 已保存到 {ego_info_path}")

                previous_close_participants_info, surrounding_actor= self.check_previous_close_participants(ego_vehicle)
                surrounding_info_path = 'D:\\Github仓库\\llm_direct_control\\scene_data\\surrounding_positions.json'
                with open(surrounding_info_path, 'w') as f:
                    json.dump(previous_close_participants_info, f, indent=4)
                    #print(f"surrounding_info 已保存到 {surrounding_info_path}")

                # 获取新的决策
                output = get_gpt_result()
                path_decision = output.path_decision
                speed_decision = output.speed_decision  
                print(f'New decision received - Path: {path_decision}, Speed: {speed_decision}')

                ego_vehicle.set_autopilot(enabled=True)
                traffic_manager = self.client.get_trafficmanager()
                traffic_manager.ignore_vehicles_percentage(ego_vehicle,1)
                
                # 验证决策是否有效
                valid_path_decisions = ['LEFT_LANE_CHANGE', 'RIGHT_LANE_CHANGE', 'LEFT_LANE_OVERTAKE', 'RIGHT_LANE_OVERTAKE', 'LANE_KEEP']
                valid_speed_decisions = ['ACCELERATE', 'DECELERATE', 'MAINTAIN', 'STOP']
                
                if path_decision in valid_path_decisions and speed_decision in valid_speed_decisions:
                    # 执行速度控制
                    speed_control = self.speed_decision_define(speed_decision)
                    if speed_control:
                        _, success = speed_control(ego_vehicle)
                        if success:
                            print("Speed command executed")

                    # 执行路径控制
                    path_control = self.path_decision_define(path_decision, ego_vehicle)
                    if path_control:
                        _, success = path_control(ego_vehicle)
                        if success:
                            print("Path command executed")

                    # 执行2秒的控制
                    execution_ticks = int(1.0 / 0.05)  # 1秒除以tick间隔0.05秒
                    for _ in range(execution_ticks):
                        self.world.tick()    
                        time.sleep(0.05)
                else:
                    print("Invalid decision input")

        except KeyboardInterrupt:
            print("User interrupted control loop")
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
        except Exception as e:
            print(f"Control process error: {e}")
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))

    def destroy_actors(self):
        """Removes the previously spawned actors (vehicles and pedestrians/walkers)"""
        # Remove vehicles
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            vehicle.destroy()

        time.sleep(1.0)

    def clear_all_vehicles(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        time.sleep(0.5)

    

 

if __name__ == '__main__':
    try:
        CARLA = CarlaWorld()
        SYNC_MODE = True
        # 确保同步模式正确设置
        settings = CARLA.world.get_settings()
        settings.synchronous_mode = SYNC_MODE
        settings.fixed_delta_seconds = 0.05
        CARLA.world.apply_settings(settings)
        
        # 生成自车
        ego_vehicle, origin = CARLA.spawn_ego_vehicle()
        time.sleep(2)
        if ego_vehicle is None:
            print("Ego vehicle spawn failed, program exiting")
            sys.exit(1)
            
        # 生成场景
        trigger_point, selected_task = CARLA.generate_tasks()
        CARLA.spawn_actors_debug(origin, selected_task)
        CARLA.generate_spectator()
        
        try:
            # 运行自车控制
            CARLA.ego_control(ego_vehicle)

            if SYNC_MODE:
                    CARLA.world.tick()
                    time.sleep(1)

        except KeyboardInterrupt:
            print("User interrupted simulation")
        except Exception as e:
            print(f"Scenario running error: {e}")
        finally:
            # 清理资源
            if 'ego_vehicle' in locals() and ego_vehicle is not None:
                ego_vehicle.destroy()
            # 清理场景
            CARLA.destroy_actors()
            settings = CARLA.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            CARLA.world.apply_settings(settings)

    except Exception as e:
        print(f"Program running error: {e}")

