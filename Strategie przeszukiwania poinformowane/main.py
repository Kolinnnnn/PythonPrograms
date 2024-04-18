import time
import ast

class Nhetman:
    def __init__(self, n):
        self.n = n
        self.open_list = []
        self.closed_list = set()

    #wyswietla rozwiazanie, przeksztalca z tekstu na liste wspolrzednych
    def print_solution(self, state):
        state_list = eval(state)
        solution = [hetman for hetman in state_list]
        print("Rozwiązanie:")
        print(solution)

    #rozwiazuje problem
    def solve(self, heuristic):
        start_time = time.time()
        initial_state = []

        if heuristic == "heuristic1":
            result = self.best_first_search(initial_state, self.heu1)
        elif heuristic == "heuristic2":
            result = self.best_first_search(initial_state, self.heu2)
        elif heuristic == "heuristic3":
            result = self.best_first_search(initial_state, self.heu3)
        else:
            print("Nieznana heurystyka.")
            result = []

        if result:
            self.print_board(result[-1])
            self.print_solution(result[-1])
            self.print_stats()
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Czas wykonania: {execution_time:.5f} sekundy")

    #algorytm bfs
    def best_first_search(self, s0, heuristic):
        h_values = {}
        parent_pointers = []

        s0 = tuple(s0)
        s0_str = str(s0)

        h_values[s0_str] = heuristic(s0)
        parent_pointers.append((s0_str, None))
        self.open_list.append(s0_str)

        while self.open_list:
            self.open_list.sort(key=lambda state: h_values[state])
            current_state = self.open_list.pop(0)

            if current_state in self.closed_list:
                continue

            self.closed_list.add(current_state)

            if self.endstate(ast.literal_eval(current_state)):
                return self.reconstruct_path(current_state, parent_pointers)

            child_states = self.generate_children(ast.literal_eval(current_state))

            for child_state in child_states:
                child_state_str = str(child_state)

                if child_state_str in self.closed_list:
                    continue

                h_values[child_state_str] = heuristic(child_state)
                parent_pointers.append((child_state_str, current_state))

                self.open_list.append(child_state_str)
                self.open_list.sort(key=lambda state: h_values[state])

        return None

    def heu1(self, state):
        Attacks = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if self.conflict_checker(state, state[i][0], state[i][1]) or self.conflict_checker(state, state[j][0], state[j][1]):
                    Attacks += 1
        return Attacks

    def heu2(self, state):
        Attacks = 0
        MidRow = len(state) // 2

        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if self.conflict_checker(state, state[i][0], state[i][1]) or self.conflict_checker(state, state[j][0], state[j][1]):
                    Attacks += 1

        Dev = [abs(hetman[0] - MidRow) for hetman in state]
        DevSum = sum(Dev)
        return DevSum

    def heu3(self, state):
        Attacks = 0

        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if self.conflict_checker(state, state[i][0], state[i][1]) or self.conflict_checker(state, state[j][0], state[j][1]):
                    Attacks += 1

        manhattan_dist = [abs(state[i][0] - state[j][0]) + abs(state[i][1] - state[j][1]) for i in range(len(state)) for j in range(i + 1, len(state))]

        sep = 3
        deviation2 = sum(abs(dist - sep) for dist in manhattan_dist)
        return deviation2
    
    #sprawdza czy wszystkie hetmany sa na planszy
    def endstate(self, s):
        return len(s) == self.n

    #generuje mozliwe ruchy
    def generate_children(self, state):
        children = []
        for row in range(self.n):
            if all(row != hetman[0] for hetman in state):
                for col in range(self.n):
                    if not self.conflict_checker(state, row, col):
                        child_state = list(state) + [(row, col)]
                        children.append(child_state)
        return children

    #odtwarza ścieżkę od stanu końcowego do stanu początkowego
    def reconstruct_path(self, current_state, parent_pointers):
        path = []
        goal_state_str = current_state
        while goal_state_str is not None:
            path.insert(0, goal_state_str)
            for parent, state in parent_pointers:
                if state == goal_state_str:
                    goal_state_str = parent
                    break
            else:
                break
        return path

    #sprawdzanie konfliktow 
    def conflict_checker(self, state, row, col):
        for hetman in state:
            if hetman[0] == row or hetman[1] == col or abs(hetman[0] - row) == abs(hetman[1] - col):
                return True
        return False

    #wyswietlanie tablicy
    def print_board(self, state):
        board = [[' ' for _ in range(self.n)] for _ in range(self.n)]
        state_list = ast.literal_eval(state)
        for hetman in state_list:
            board[hetman[0]][hetman[1]] = 'Q'
    
        for row in board:
            print("+---" * self.n + "+")
            for cell in row:
                print("| " + cell + " ", end="")
            print("|")
        print("+---" * self.n + "+")

    #statystyki
    def print_stats(self):
        print(f"Długość listy open: {len(self.open_list)}")
        print(f"Długość listy closed: {len(self.closed_list)}")

solver = Nhetman(4)
solver.solve("heuristic3")