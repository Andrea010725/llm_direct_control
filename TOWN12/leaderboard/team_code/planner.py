import os
from collections import deque
import math
import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        # self.mean = np.array([0.0, 0.0]) # for carla 9.10
        # self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

        # self.mean = np.array([35.25000000664444, -101.8749994504694])
        # self.scale = np.array([111324.60662786, 90908.12207377452]) # for town12

        self.lon_ref = -101.8749994504694
        self.lat_ref = 35.25000000664444

        #if DEBUG:
        #    self.debug = Plotter(debug_size)
 
    def _gps_to_loc(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        lat = gps[0]
        lon = gps[1]

        x = (lon - self.lon_ref) * scale * math.pi * EARTH_RADIUS_EQUA / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0) / math.tan((90.0 + lat) * math.pi / 360.0))

        return np.array([-y, x])

    def set_route(self, global_plan, gps=False, global_plan_world = None):
        self.route.clear()

        if global_plan_world:
            for (pos, cmd), (pos_word, _ )in zip(global_plan, global_plan_world):
                if gps:
                    pos = np.array([pos['lat'], pos['lon']])
                    pos = self._gps_to_loc(pos)
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean
                
                self.route.append((pos, cmd, pos_word))
        else:
            for pos, cmd in global_plan:
                if gps:
                    pos = np.array([pos['lat'], pos['lon']])
                    pos = self._gps_to_loc(pos)
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean

                self.route.append((pos, cmd))

    def run_step(self, gps):

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]
