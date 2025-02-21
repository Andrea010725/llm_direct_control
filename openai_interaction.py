from openai import OpenAI
import json
import math

# 生成向量的模板
vector_template = """
    Background:
    You are a driving assistant responsible for controlling the vehicle. Based on the driving conditions, you need to select a path decision and a speed decision from predefined options, ensuring no collision with other moving or stationary objects.
    
    Please analyze the current driving scenario using data from JSON files:
    Ego vehicle position: ({ego_position_x}, {ego_position_y})
    Ego vehicle velocity: ({ego_velocity_x}, {ego_velocity_y})
    Surrounding environment information:
    {surrounding_info}
    When {overall_safety_flag} is 1, it indicates the current scenario needs extra attention and requires reasonable path and speed decisions.

    Important rules:
    - Once a stationary obstacle is found ahead, directly! use the lane change plan. When changing lanes, consider the positions and speeds of surrounding traffic participants to select a safe path.
    - If there is a slow-moving vehicle ahead, directly!!!! choose LEFT_LANE_CHANGE as the path decision and keep lane 
    - Avoid frequent lane changes  !!!!!!!
    - Three consecutive lane change decisions are not allowed

    Path Decision Definitions:
    "LEFT_LANE_CHANGE" means the driver decides to switch from current lane to the adjacent left lane.
    "RIGHT_LANE_CHANGE" means the driver decides to switch from current lane to the adjacent right lane.
    "LEFT_LANE_OVERTAKE" means the driver temporarily uses the adjacent left lane, usually for overtaking or avoiding obstacles.
    "RIGHT_LANE_OVERTAKE" means the driver temporarily uses the adjacent right lane, usually for overtaking or avoiding obstacles.
    "LANE_KEEP" means the driver decides to maintain the current lane.
    
    Speed Decision Definitions:
    "ACCELERATE" means the driver increases vehicle speed.
    "DECELERATE" means the driver decreases vehicle speed.
    "MAINTAIN" means the driver maintains current speed.
    "STOP" means the driver completely stops the vehicle.
    
    Based on the path decision definitions and traffic rules, please select path and speed decisions from the predefined options:
    Path decisions include [LEFT_LANE_CHANGE, RIGHT_LANE_CHANGE, LEFT_LANE_OVERTAKE, RIGHT_LANE_OVERTAKE, LANE_KEEP].
    Speed decisions include [ACCELERATE, DECELERATE, MAINTAIN, STOP].
    
    Question: {question}
    Please select path and speed decisions based on my requirements, with the following requirements:
    - Both decisions must be chosen from the previous definitions
    - Output decisions directly without additional questions
    - Path decision must be wrapped in <PATHVECTOR>, speed decision must be wrapped in <SPEEDVECTOR>

    Please output only these two vectors without any additional content
    """



# 分析向量的模板
analysis_template = """
Background: You are a driving assistant responsible for controlling the vehicle. You need to analyze the reasonableness of the current path decision and speed decision.

Ego vehicle position: {ego_position}
Ego vehicle coordinates: {ego_position_x}
Surrounding environment information: {surrounding_info}
Current safety flag: {overall_safety_flag}
Once a stationary obstacle is found ahead, directly use the lane change plan. When changing lanes, consider the positions and speeds of surrounding traffic participants to select a safe path.
If the vehicle in front is extremely slow, directly use the left lane change plan. After passing the slow vehicle, accelerate.
If there is a slow-moving vehicle ahead, directly choose LEFT_LANE_CHANGE as the path decision ！！！
Behavior does not need to be very conservative. You can pursue efficiency to some extent.
For different surrounding traffic participants, you can use different path and speed decision plans.
After changing lanes, pay attention not to have continuous lane changes or frequent lane changes.
Pay attention not to collide, and you can consider efficiency under the premise of ensuring safety.
System decision:
Path decision: {path_decision}
Speed decision: {speed_decision}

Please analyze the reasonableness of this decision and explain how this decision balances safety, efficiency, and comfort.
"""

# 问题是: {question}，请根据我的需求。分析一下决策结果的合理性
def calculate_distance(ego_position, other_position):
    """计算自车与周围车辆之间的距离"""
    return math.sqrt((ego_position['x'] - other_position['x']) ** 2 +
                    (ego_position['y'] - other_position['y']) ** 2)

def calculate_speed(velocity):
    """计算车辆的速度大小"""
    return math.sqrt(velocity['x'] ** 2 + velocity['y'] ** 2)

def calculate_ttc(ego_position, ego_velocity, other_position, other_velocity):
    """计算TTC（时间到碰撞）"""
    distance = calculate_distance(ego_position, other_position)
    relative_velocity_x = ego_velocity['x'] - other_velocity['x']
    relative_velocity_y = ego_velocity['y'] - other_velocity['y']
    relative_velocity = math.sqrt(relative_velocity_x ** 2 + relative_velocity_y ** 2)
    
    return float('inf') if relative_velocity == 0 else distance / relative_velocity

def generate_surroundinginfo_template(json_path, ego_position, ego_velocity):
    """生成周围交通参与者信息模板"""
    surrounding_info = read_positions(json_path)
    surrounding_positions = []
    overall_safety_flag = 0
    
    for idx, participant in enumerate(surrounding_info):
        other_position = participant['location']
        other_velocity = participant.get('velocity', {'x': 0, 'y': 0})
        ttc = calculate_ttc(ego_position, ego_velocity, other_position, other_velocity)
        

        # 安全检查逻辑
        if ((participant['type'].startswith('vehicle') and participant['is_in_front'] and ttc < 10) or
            (participant['type'].startswith('walker') and participant['distance'] < 20) or
            (participant['type'].startswith('static') and participant['distance'] < 25)):
            overall_safety_flag = 1

        participant_info = (
            f"The type of the {idx + 1}th surrounding traffic participant is: {participant['type']}, "
            f"position is ({participant['location']['x']}, {participant['location']['y']}), "
            f"speed is ({other_velocity['x']}, {other_velocity['y']}), "
            f"whether in front: {participant['is_in_front']}, "
            f"distance to ego vehicle is {participant['distance']:.2f} meters"
            f"TTC = {ttc:.2f} seconds, "
            f"safety weight = {overall_safety_flag}"
        )
        surrounding_positions.append(participant_info)

    return surrounding_positions, overall_safety_flag

client = OpenAI(
    # api_key="sk-8578ff9997934be48b142b5996a47927", # API Key替换
    api_key="sk-ad06a4a820324306961309ad635f0a79",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url

)

def generate_response(prompt: str) -> str:
    """ 使用 OpenAI API 生成模型响应 """
    response = client.chat.completions.create(
        model="qwen-turbo",     #"deepseek-r1-distill-llama-8b"  "deepseek-v3" "deepseek-r1"  "qwen-turbo"   qwen-plus"换成一个小模型
        messages=[
             {'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user',
              'content': prompt}])

    return response.choices[0].message.content

def query_vector(message: str, ego_position: str, ego_position_x: float, ego_position_y: float
                 , ego_velocity_x: float, ego_velocity_y: float, surrounding_info: str, overall_safety_flag: float) -> str:
    """ Generate vector """
    prompt = vector_template.format(question=message,
                                    ego_position=ego_position,
                                    ego_position_x=ego_position_x,
                                    ego_position_y=ego_position_y,
                                    ego_velocity_x=ego_velocity_x,
                                    ego_velocity_y=ego_velocity_y,
                                    surrounding_info=surrounding_info,
                                    overall_safety_flag=overall_safety_flag
                                    )
    response = generate_response(prompt)
    
    # 添加结果验证和清理
    if '<PATHVECTOR>' in response and '<SPEEDVECTOR>' in response:
        # 提取路径和速度决策
        path_start = response.find('<PATHVECTOR>') + len('<PATHVECTOR>')
        path_end = response.find('</PATHVECTOR>')
        speed_start = response.find('<SPEEDVECTOR>') + len('<SPEEDVECTOR>')
        speed_end = response.find('</SPEEDVECTOR>')
        
        if all(x != -1 for x in [path_start, path_end, speed_start, speed_end]):
            path_decision = response[path_start:path_end].strip()
            speed_decision = response[speed_start:speed_end].strip()
            
            # 构建标准格式的响应
            return f"<PATHVECTOR>{path_decision}</PATHVECTOR>\n<SPEEDVECTOR>{speed_decision}</SPEEDVECTOR>"
    
    # 如果解析失败，返回默认安全决策
    return "<PATHVECTOR>LANE_KEEP</PATHVECTOR>\n<SPEEDVECTOR>DECELERATE</SPEEDVECTOR>"

def analyze_vector(vector: str, message: str, ego_position: str, ego_position_x: float, surrounding_info: str, overall_safety_flag: float) -> str:
    """分析向量，包含位置信息"""
    # 从vector中提取实际的决策
    path_start = vector.find('<PATHVECTOR>') + len('<PATHVECTOR>')
    path_end = vector.find('</PATHVECTOR>')
    speed_start = vector.find('<SPEEDVECTOR>') + len('<SPEEDVECTOR>')
    speed_end = vector.find('</SPEEDVECTOR>')
    
    path_decision = vector[path_start:path_end].strip()
    speed_decision = vector[speed_start:speed_end].strip()
    
    # 更新模板以使用实际的决策
    analysis_template = """
    Background: You are a driving assistant responsible for controlling the vehicle. You need to analyze the reasonableness of the current path decision and speed decision.
    
    Ego vehicle position: {ego_position}
    Ego vehicle coordinates: {ego_position_x}
    Surrounding environment information: {surrounding_info}
    Current safety flag: {overall_safety_flag}
    
    System decision:
    Path decision: {path_decision}
    Speed decision: {speed_decision}
    
    Please analyze the reasonableness of this decision and explain how this decision balances safety, efficiency, and comfort.
    """
    
    prompt = analysis_template.format(
        ego_position=ego_position,
        ego_position_x=ego_position_x,
        surrounding_info=surrounding_info,
        overall_safety_flag=overall_safety_flag,
        path_decision=path_decision,
        speed_decision=speed_decision
    )

    return generate_response(prompt)

def read_positions(file_path: str):
    """从文件中读取位置信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        positions = json.load(file)
    return positions

def get_key_from_json(file_path: str, location_key: str) -> float:          # E.G.  输出到position
    """ 从JSON文件中读取数据，并根据给定的location_key和value_key获取对应的值 """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get(location_key, {})


def get_value_from_json(file_path: str, location_key: str, value_key: str) -> float:   # e.g. 输出的到 position.x
    """ 从JSON文件中读取数据，并根据给定的location_key和value_key获取对应的值 """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get(location_key, {}).get(value_key)


def get_response(message: str):
    """获取向量及其分析"""
    base_path = 'D:\\Github仓库\\llm_direct_control\\scene_data'
    
    # 读取场景数据
    #scene_flag = read_positions(f'{base_path}\\scene_flag.json')
    ego_position = get_key_from_json(f'{base_path}\\ego_position.json', 'vehicle_location')
    ego_velocity = get_key_from_json(f'{base_path}\\ego_position.json', 'vehicle_velocity')
    
    # 获取具体位置和速度值
    ego_position_x = ego_position['x']
    ego_position_y = ego_position['y']
    ego_velocity_x = ego_velocity['x']
    ego_velocity_y = ego_velocity['y']
    
    # 生成周围信息
    surrounding_info, overall_safety_flag = generate_surroundinginfo_template(
        f'{base_path}\\surrounding_positions.json',
        ego_position,
        ego_velocity
    )

    # 生成向量和解释
    vector = query_vector(message, ego_position, ego_position_x, ego_position_y,
                         ego_velocity_x, ego_velocity_y, surrounding_info, overall_safety_flag)
    
    # 确保vector是标准格式
    if not (('<PATHVECTOR>' in vector) and ('<SPEEDVECTOR>' in vector)):
        vector = "<PATHVECTOR>LANE_KEEP</PATHVECTOR>\n<SPEEDVECTOR>DECELERATE</SPEEDVECTOR>"
    
    explanation = analyze_vector(vector, message, ego_position, 
                               ego_position_x, surrounding_info, overall_safety_flag)

    return [vector, explanation]

def run_server():
    """启动服务器并处理请求"""
    message = "要求车辆又考虑安全性，又考虑舒适性和效率性"
    vector, explanation = get_response(message)

    print(f"<VECTOR>{vector}</VECTOR>")
    print(f"<EXPLANATION>{explanation}</EXPLANATION>")


if __name__ == "__main__":
    run_server()