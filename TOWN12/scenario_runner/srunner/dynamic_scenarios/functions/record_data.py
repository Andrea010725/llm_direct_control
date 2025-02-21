import sys
sys.path.append("D:/Github仓库/CarlaReplay_zhiwen/")
from automatic_gpt import monitor_script
import json
import math
import carla

class RecordData(object):

    def record_data(ego_vehicle, wp,world,other_actors):

        vehicle_location = ego_vehicle.get_location()
        is_at_traffic_light = float(ego_vehicle.is_at_traffic_light())

        vehicle_velocity = ego_vehicle.get_velocity()

        # 存json文件 用来传给gpt用
        # 创建一个字典来存储 ego 信息
        ego_info = {
            "vehicle_location": {
                "x": vehicle_location.x,
                "y": vehicle_location.y,
                "z": vehicle_location.z
            },
            "vehicle_velocity": {
                "x": vehicle_velocity.x,
                "y": vehicle_velocity.y,
                "z": vehicle_velocity.z
            }
        }

        # 将 ego_info 字典转换为 JSON 格式并保存到文件
        # ego_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/ego_position.json"  # JSON 文件路径
        ego_info_path = "D:\\Github仓库\\CarlaReplay_zhiwen\\scene_data\\ego_position.json"
        with open(ego_info_path, 'w') as json_file:
            json.dump(ego_info, json_file, indent=4)  # 美化输出，缩进为 4 个空格
            #print("ego_position的json文件已经更新！！！！")

        # ipdb.set_trace()

        scene_info = {
            "scene_type": {
                "waypoint.is_intersection": float(wp.is_intersection),
                "waypoint.is_junction": float(wp.is_junction),
                "is_at_traffic_light": is_at_traffic_light
            }
        }
        # scene_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/scene_flag.json"  # JSON 文件路径
        scene_info_path = "D:\\Github仓库\\CarlaReplay_zhiwen\\scene_data\\scene_flag.json"
        with open(scene_info_path, 'w') as json_file:
            json.dump(scene_info, json_file, indent=4)  # 美化输出，缩进为 4 个空格

        # # 周围交通参与者的信息
        hero_vehicle = ego_vehicle

        previous_close_vehicles_info, surrounding_vehicles = RecordData.check_surrounding_vehicles(world,ego_vehicle,other_actors)
       # print("previous_close_vehicles_info", previous_close_vehicles_info)

        # 将信息存储到 JSON 文件中
        # surrounding_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/surrounding_positions.json"
        surrounding_info_path = "D:\\Github仓库\\CarlaReplay_zhiwen\\scene_data\\surrounding_positions.json"
        with open(surrounding_info_path, 'w') as json_file:
            json.dump(previous_close_vehicles_info, json_file, indent=4)

    def check_surrounding_vehicles(world, ego_vehicle, other_actors):
        """
        检查自车周围的所有车辆、静态物体和行人信息。

        :param world: CARLA 世界对象
        :param ego_vehicle: 自车 Actor
        :param other_actors: 其他参与者列表
        :return: surrounding_info 包含周围车辆、静态物体和行人的信息
                 surrounding_actors 包含所有周围参与者 Actor
        """
        surrounding_info = []
        surrounding_actors = []
        N = 0

        if ego_vehicle is None:
            print("无法找到自车 (ego_vehicle)！")
            return surrounding_info, surrounding_actors

        # 获取自车位置
        hero_location = ego_vehicle.get_location()

        # 遍历所有 Actor
        for actor in world.get_actors():
            # 如果是自车，跳过
            if actor.id == ego_vehicle.id:
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

            # 仅记录距离在 20 米以内的 Actor
            if distance <= 20:
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
                if RecordData.is_in_front_of_hero_vehicle(ego_vehicle, actor):
                    actor_info['is_in_front'] = True
                else:
                    actor_info['is_in_front'] = False

                surrounding_info.append(actor_info)

                # Debug 信息
                if N == 0:
                   # print(f"当前场景中共有 {len(world.get_actors())} 个 Actor！")
                    N += 1

        return surrounding_info, surrounding_actors

    # def check_surrounding_vehicles(world,ego_vehicle,other_actors):
    #     surrounding_info = []
    #     surrounding_vehicles = []
    #     N = 0
    #     # while True:
    #     if ego_vehicle is None:
    #         print("can not find ego_vehicle!!!")
    #     else:
    #         #print("find ego_vehicle!!!")
    #         hero_location = ego_vehicle.get_location()
    #         for vehicle in world.get_actors().filter('vehicle.*'):
    #             surrounding_vehicles.append(vehicle)
    #             if N == 0:
    #                 #print(f"there are  { len(self._world.get_actors().filter('vehicle.*'))} Vehicles in the Town!!!")
    #                 N = N + 1
    #
    #             if vehicle.id == ego_vehicle.id:
    #                 continue
    #
    #             else:
    #                 # ipdb.set_trace()
    #                 vehicle_location = ego_vehicle.get_location()
    #                 vehicle_velocity = ego_vehicle.get_velocity()
    #                 distance1 = math.sqrt(
    #                     (hero_location.x - vehicle_location.x) ** 2 +
    #                     (hero_location.y - vehicle_location.y) ** 2
    #                 )
    #                 if distance1 <= 20:  # 距离20米之内
    #                     for vehicle in other_actors:
    #                         if RecordData.is_in_front_of_hero_vehicle(ego_vehicle,vehicle):
    #                          #   print("detecting One Vehicle satisfying the in_front_of_hero_vehicle function!!!")
    #                           #  print("!!!!!!  Other_Vehicle_velocity", vehicle_velocity)
    #                             surrounding_info.append({
    #                                 'location': {
    #                                     'x': vehicle_location.x,
    #                                     'y': vehicle_location.y,
    #                                     'z': vehicle_location.z
    #                                 },
    #                                 'velocity': {
    #                                     'x': vehicle_velocity.x,
    #                                     'y': vehicle_velocity.y,
    #                                     'z': vehicle_velocity.z
    #                                 },
    #                                 'distance': [distance1]
    #                             })
    #    # print("Finished checking surrounding vehicles")
    #     # print("surrounding_info", surrounding_info)
    #     # print("surrounding_vehicles", surrounding_vehicles)
    #     return surrounding_info, surrounding_vehicles

        # 判断是否在自车前面  （带朝向的）

    def is_in_front_of_hero_vehicle(ego_vehicle, other_vehicle):
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



    def _monitor_loop():
        """
        监控脚本的循环逻辑，每隔1秒执行 monitor_script()
        """

        try:
            monitor_script()
            print("monitor_script running")
        except Exception as e:
            print(f"Error in monitor_script: {e}")