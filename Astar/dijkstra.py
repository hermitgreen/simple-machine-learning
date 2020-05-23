# -*- coding: utf-8 -*-
import math
tm = [
    '##########################################',
    '#.....................................#..#',
    '#.....................................#..#',
    '#.....................................####',
    '#####....................................#',
    '#...#..........S.........................#',
    '#...#####................................#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '#...................#....................#',
    '#...................#....................#',
    '#...................#....................#',
    '#.#########..#......#..........####......#',
    '#.#..........#......#..........#####.....#',
    '#.#..........#......#..........#####.....#',
    '#.#..........#......#..........#.........#',
    '#.#########..########..........#.........#',
    '#..............................#.........#',
    '#...................##.........#.........#',
    '#...................############.........#',
    '#...................################.....#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '#.......######...........................#',
    '#.......#....#...........................#',
    '#.......#....#...........................#',
    '#.......#....#....######.................#',
    '#.......#....#.........#.................#',
    '#.......#....#.........#.................#',
    '#.......#....#....######.................#',
    '#.......#....#...........................#',
    '#.......######...........................#',
    '#........................E...............#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '#........................................#',
    '##########################################']
test_map = []


class Node_Elem:
    def __init__(self, parent, x, y, dist):
        self.parent = parent
        self.x = x
        self.y = y
        self.dist = dist


class DIJ:
    def __init__(self, s_x, s_y, e_x, e_y, w=42, h=42):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y
        self.width = w
        self.height = h
        self.open = []
        self.close = []
        self.path = []
        self.min = {}

    def find_path(self):
        # 构建开始节点
        p = Node_Elem(None, self.s_x, self.s_y, 0.0)
        flag = True
        while True:
            # 扩展F值最小的节点
            self.extend_round(p)
            # 更新当前的最小值
            self.update_min(p)
            # 如果开放列表为空，则不存在路径，返回
            if not self.open:
                return
            # 获取F值最小的节点
            idx, p = self.get_best()
            # 找到路径，生成路径，继续处理
            if self.is_target(p) and flag:
                self.make_path(p)
                flag = False
            # 把此节点压入关闭列表，并从开放列表里删除
            self.close.append(p)
            del self.open[idx]

    # 更新节点的最小值
    def update_min(self, p):
        idx = (p.x, p.y)
        now = self.min.get(idx, None)
        # 如果字典里没有当前节点,就插入这个节点
        if now == None:
            self.min[idx] = p
        # 否则只有当当前节点的距离更大时,才更新这个节点
        elif now.dist > p.dist:
            self.min[idx] = p
        return

    # 找坐标为x和y的节点的最短路
    def make_path_for_any_node(self, x, y):
        p = self.min[(x, y)]
        self.path.clear()
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    # 找坐标为x和y的节点的最短路长度
    def get_cost_for_any_node(self, x, y):
        p = self.min[(x, y)]
        return p.dist

    def make_path(self, p):
        while p:
            self.path.append((p.x, p.y))
            p = p.parent

    def is_target(self, i):
        return i.x == self.e_x and i.y == self.e_y

    def get_best(self):
        best = None
        bv = 1000000  # 如果你修改的地图很大，可能需要修改这个值
        bi = -1
        for idx, i in enumerate(self.open):
            value = self.get_dist(i)  # 获取F值
            if value < bv:  # 比以前的更好，即F值更小
                best = i
                bv = value
                bi = idx
        return bi, best

    def get_dist(self, i):  # 不再需要前向估值
        return i.dist

    def extend_round(self, p):
        # 可以从8个方向走
        xs = (-1, 0, 1, -1, 1, -1, 0, 1)
        ys = (-1, -1, -1, 0, 0, 1, 1, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            # 无效或者不可行走区域，则忽略
            if not self.is_valid_coord(new_x, new_y):
                continue
            # 构造新的节点
            node = Node_Elem(p, new_x, new_y, p.dist + self.get_cost(
                p.x, p.y, new_x, new_y))
            # 新节点在关闭列表，则忽略
            if self.node_in_close(node):
                continue
            i = self.node_in_open(node)
            if i != -1:
                # 新节点在开放列表
                if self.open[i].dist > node.dist:
                    # 现在的路径到比以前到这个节点的路径更好~
                    # 则使用现在的路径
                    self.open[i].parent = p
                    self.open[i].dist = node.dist
                continue
            self.open.append(node)

    def get_cost(self, x1, y1, x2, y2):
        if x1 == x2 or y1 == y2:
            return 1.0
        return 1.4  # CHE 1 L1 2

    def node_in_close(self, node):
        for i in self.close:
            if node.x == i.x and node.y == i.y:
                return True
        return False

    def node_in_open(self, node):
        for i, n in enumerate(self.open):
            if node.x == n.x and node.y == n.y:
                return i
        return -1

    def node_in_map_close(self, node):
        if(self.map_close[node.x][node.y] != 0):
            return self.map_close[node.x][node.y]
        else:
            return -1

    def node_in_map_open(self, node):
        if(self.map_open[node.x][node.y] != 0):
            return self.map_open[node.x][node.y]
        else:
            return -1

    def is_valid_coord(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return test_map[y][x] != '#'

    def get_searched(self):
        l = []
        for i in self.open:
            l.append((i.x, i.y))
        for i in self.close:
            l.append((i.x, i.y))
        return l


def print_test_map():
    for line in test_map:
        print(''.join(line))


def get_start_XY():
    return get_symbol_XY('S')


def get_end_XY():
    return get_symbol_XY('E')


def get_symbol_XY(s):
    for y, line in enumerate(test_map):
        try:
            x = line.index(s)
        except:
            continue
        else:
            break
    return x, y


def mark_path(l):
    mark_symbol(l, '*')


def mark_searched(l):
    mark_symbol(l, ',')


def mark_symbol(l, s):
    for x, y in l:
        test_map[y][x] = s


def mark_start_end(s_x, s_y, e_x, e_y):
    test_map[s_y][s_x] = 'S'
    test_map[e_y][e_x] = 'E'


def tm_to_test_map():
    for line in tm:
        test_map.append(list(line))


def find_path():
    s_x, s_y = get_start_XY()
    e_x, e_y = get_end_XY()
    a_star = DIJ(s_x, s_y, e_x, e_y)
    a_star.find_path()
    l = a_star.get_cost_for_any_node(20, 5)
    print("path length of 20 5 is %d" % l)
    searched = a_star.get_searched()
    path = a_star.path
    # 标记已搜索区域
    mark_searched(searched)
    # 标记路径
    mark_path(path)
    print("path length is %d" % (len(path)))
    print("searched squares count is %d" % (len(searched)))
    # 标记开始、结束点
    mark_start_end(s_x, s_y, e_x, e_y)


if __name__ == "__main__":
    # 把字符串转成列表
    tm_to_test_map()
    find_path()
    print_test_map()
