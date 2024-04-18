import queue
import time

class Nhetman:
    def __init__(self, n):
        self.n = n
        self.open_list = queue.Queue()
        self.closed_list = set()

    def print_solution(self, state):
        state_list = eval(state)
        solution = [hetman for hetman in state_list]
        print("Rozwiązanie:")
        print(solution)

    def solve(self, algorithm):
        start_time = time.time()
        if algorithm == "DFS":
            initial_state = []
            result = self.dfs(initial_state)
        elif algorithm == "BFS":
            initial_state = []
            result = self.bfs(initial_state)
        else:
            print("Nieznany algorytm.")
            result = []
        if result:
            self.print_board(result[-1])
            self.print_solution(result[-1])
            self.print_stats()
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Czas wykonania: {execution_time:.5f} sekundy")

    def bfs(self, s0):
        parent = {}
        self.open_list.put(tuple(s0))
        while not self.open_list.empty():
            s = self.open_list.get()
            state_str = str(s)
            if self.endstate(s):
                return self.reconstruct_path(parent, s)

            self.closed_list.add(state_str)
            children = self.generate_children(s)

            for t in children:
                t_str = str(t)
                if t_str not in self.closed_list and t not in list(self.open_list.queue):
                    self.open_list.put(t)
                    parent[t_str] = state_str

        return []

    def dfs(self, s0):
        parent = {}
        self.open_list.put(tuple(s0))
        while not self.open_list.empty():
            s = self.open_list.get()
            state_str = str(s)
            if self.endstate(s):
                return self.reconstruct_path(parent, s)

            self.closed_list.add(state_str)
            children = self.generate_children(s)

            for t in children:
                t_str = str(t)
                if t_str not in self.closed_list and t not in list(self.open_list.queue):
                    self.open_list.put(t)
                    parent[t_str] = state_str

        return []
    
    def reconstruct_path(self, parent, goal_state):
        path = []
        goal_state_str = str(goal_state)
        while goal_state_str in parent:
            path.insert(0, goal_state_str)
            goal_state_str = parent[goal_state_str]
        return path

    def endstate(self, s):
        return len(s) == self.n

    def conflict_checker(self, state, row, col):
        for hetman in state:
            if hetman[0] == row or hetman[1] == col or abs(hetman[0] - row) == abs(hetman[1] - col):
                return True
        return False

    def generate_children(self, state):
        children = []
        for col in range(self.n):
            for row in range(self.n):
                if not self.conflict_checker(state, row, col):
                    child_state = tuple(list(state) + [[row, col]])
                    children.append(child_state)
        return children

    def print_board(self, state):
        board = [[' ' for _ in range(self.n)] for _ in range(self.n)]
        state_list = eval(state)
        for hetman in state_list:
            board[hetman[0]][hetman[1]] = 'Q'
    
        for row in board:
            print("+---" * self.n + "+")
            for cell in row:
                print("| " + cell + " ", end="")
            print("|")
        print("+---" * self.n + "+")

    def print_stats(self):
        print(f"Długość listy open: {self.open_list.qsize()}")
        print(f"Długość listy closed: {len(self.closed_list)}")

solver = Nhetman(4)
solver.solve("DFS")

