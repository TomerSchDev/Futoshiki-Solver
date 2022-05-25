import random
import matplotlib.pyplot as plt
import os
import pygame
import timeit

DEFAULT_MUTATIONS_PERCENTS = .05
DEFAULT_MAXIMUM_LOCAL = 25
DEFAULT_TIME_RESTARTING = 5
HEIGHT = 900
WIDTH = 800
pygame.init()


class Board:
    """
    this class represents our board that we want to solve
    """

    def __init__(self, matrix_parameters):
        """
        init the board
        :param matrix_parameters: dictionary with the instruction of the board
        """
        self.size = matrix_parameters["k"]
        self.to_place = lambda x: 2 * (x - 1)
        self.matrix = []
        self.conditions_characters = ['<', '>', '\/', '/\\']
        self.conditions = []
        self.digit_font = pygame.font.SysFont('comicsans', 40)
        self.con_font = pygame.font.SysFont('Arial', 50)
        # lambda to check if a condition is happening
        self.check_condition = lambda gy, gx, ly, lx, c_matrix: c_matrix[gy][gx] > c_matrix[ly][lx]
        for _ in range(self.size):
            row_n = []
            for i in range(self.size):
                row_n.append(0)
            self.matrix.append(row_n)
        self.add_digits(matrix_parameters["digits"])
        self.add_conditions(matrix_parameters["conditions"])

    def draw(self, screen, matrix, gen, mutations_matrix):
        """
        this function draw the solver to the screen
        :param screen: the screen
        :param matrix: the solution we want to draw
        :param gen:what number of generation we are
        :param mutations_map: what places are need to change
        :return:
        """
        pygame.Surface.fill(screen, 'burlywood2')
        gen_text = self.con_font.render(f"Generation :{gen}", True, 'black')
        screen.blit(gen_text, ((WIDTH - gen_text.get_width()) // 2, 20))
        start_y = 20 + gen_text.get_height()
        # calculate how big the blocks going to be based on how many we needs
        block_size = WIDTH // (self.size * 2)
        for i in range(self.size):
            for j in range(self.size):
                y = start_y + i * 2 * block_size + block_size // 2
                x = j * 2 * block_size + block_size // 2
                pygame.draw.rect(screen, 'black', [x, y, block_size, block_size], 1, 1)
                color = 'black'
                # if it is a pixed digit draw it red
                if self.matrix[i][j] != 0:
                    color = 'blue'
                elif mutations_matrix is not None and mutations_matrix[i][j] != 0:
                    color = 'red'
                digit_text = self.digit_font.render(str(matrix[i][j]), True, color)
                screen.blit(digit_text, (
                    x + block_size // 2 - digit_text.get_width() // 2,
                    y + block_size // 2 - digit_text.get_height() // 2))
        for (gy, gx, ly, lx), c in self.conditions:
            mx = (lx + gx) / 2
            my = (ly + gy) / 2
            y = start_y + my * 2 * block_size + block_size * 1 / 2
            x = mx * 2 * block_size + block_size * 3 / 4
            conditions_text = self.con_font.render(self.conditions_characters[c], True, 'black')
            screen.blit(conditions_text, (x, y))
        pygame.display.update()

    def is_solved(self, solution):
        """
        this function checks if a given solution solves the board
        :param solution: the solution to check
        :return: if the board is solved
        """
        # add the solution to the board
        copy_matrix = self.solution_matrix(solution)
        for (gy, gx, ly, lx), c in self.conditions:
            # check if all the conditions are taking place
            if not self.check_condition(gy, gx, ly, lx, copy_matrix):
                return False
        # check if solution obey limit numbers in rows
        for i in range(self.size):
            check = [0 for _ in range(self.size)]
            for j in range(self.size):
                digit = copy_matrix[i][j] - 1
                check[digit] += 1
            if max(check) != 1:
                return False
        # check if solution obey limit numbers in cols
        for i in range(self.size):
            check = [0 for _ in range(self.size)]
            for j in range(self.size):
                digit = copy_matrix[j][i] - 1
                check[digit] += 1
            if max(check) != 1:
                return False
        # if the solution obey every thing then the board is solved
        return True

    def add_digits(self, digits):
        """
        this function takes the list of digits given in the config file and place them in the right place
        :param digits:list of digits
        :return:nothing
        """
        for yt, xt, d in digits:
            x = xt - 1
            y = yt - 1
            self.matrix[y][x] = d

    def add_conditions(self, conditions):
        """
        this function takes the list of conditions given in the config file and place them in the right place
        :param conditions: the list of conditions
        :return: nothing
        """
        for y1, x1, y2, x2 in conditions:
            gx = x1 - 1
            lx = x2 - 1
            gy = y1 - 1
            ly = y2 - 1
            # calculate the middle place to place the character of the condition
            ty = (gy + ly) // 2
            tx = (gx + lx) // 2
            # check what character to put
            if lx == gx:
                if ly < gy:
                    c = 3
                else:
                    c = 2
            else:
                if lx < gx:
                    c = 0
                else:
                    c = 1
            # appand the condition to the list
            self.conditions.append(((y1 - 1, x1 - 1, y2 - 1, x2 - 1), c))

    def print(self, matrix=None):
        """
        print the board
        :param matrix: if given it print the board with the solution matrix if not it will print empty board
        :return:
        """
        if matrix is None:
            # if not given solution matrix print with the board matrix
            matrix = self.matrix
        top = ""
        for _ in range(4 * self.size - 1):
            top += "-"
        print(top)
        for i, row in enumerate(matrix):
            string = "|"
            for j, c in enumerate(row):
                digit = str(c)
                if c == 0:
                    # digit = " "
                    pass
                switch_j = 0
                write = digit
                if i % 2 == 1:
                    switch_j = 1
                    write = 'X'
                if j % 2 == switch_j:
                    string += write
                else:
                    string += self.conditions_characters[c]
                string += "|"
            print(string)
        print(top)

    def draw_solution(self, win, solution, gen, mutations_map=None):
        """
        this function will print the board with a given solution
        :param solution: the solution to draw
        :return: nothing
        """
        solution_matrix = self.solution_matrix(solution)
        self.draw(win, solution_matrix, gen, mutations_map)

    def complete_solution_matrix(self, solution_matrix):
        """
        this function get a solution matrix and turn it into a board matrix (with conditions character)
        :param solution_matrix: the solution matrix
        :return:complere matrix_parameters of the solution
        """
        matrix = []
        for i, row in enumerate(self.matrix):
            new_row = []
            for j, c in enumerate(row):
                if i % 2 == 1:
                    new_row.append(c)
                elif j % 2 == 1:
                    new_row.append(c)
                else:
                    x = j // 2
                    y = i // 2
                    new_row.append(solution_matrix[y][x])
            matrix.append(new_row)
        return matrix

    def solution_matrix(self, solution):
        """
        this function turn a solution (of type list) to a matrix with the digits on the board
        :param solution: the given solution
        :return: matrix representing the solution
        """
        index = 0
        matrix = []
        for y in range(self.size):
            n_row = []
            for x in range(self.size):
                d = self.matrix[y][x]
                if d == 0:
                    d = solution[index]
                    index += 1
                n_row.append(d)
            matrix.append(n_row)
        return matrix

    def evaluate_solution(self, solution):
        """
        this function get a solution and return its fitness score
        :param solution: the solution to check
        :return: the fitness score
        """
        copy_matrix = self.solution_matrix(solution)
        fitness_score = 0
        # how many points each section gets
        points_for_conditions = 70
        points_for_row = 15
        points_for_col = 15

        # check if solution obey conditions
        for (gy, gx, ly, lx), c in self.conditions:
            if self.check_condition(gy, gx, ly, lx, copy_matrix):
                fitness_score += points_for_conditions / len(self.conditions)
        # check if solution obey limit numbers in rows
        for i in range(self.size):
            check = [0] * self.size
            y = i
            for j in range(self.size):
                x = j
                digit = copy_matrix[y][x] - 1
                check[digit] += 1
            # if all the digits are only shown once
            if max(check) == 1:
                fitness_score += points_for_row / self.size
        # check if solution obey limit numbers in cols
        for i in range(self.size):
            check = [0] * self.size
            x = i
            for j in range(self.size):
                y = j
                digit = copy_matrix[y][x] - 1
                check[digit] += 1
            if max(check) == 1:
                fitness_score += points_for_col / self.size
        return fitness_score

    def get_mutations_matrix(self, solution):
        copy_matrix = self.solution_matrix(solution)
        optim_matrix = [[0 for _ in range(self.size)] for i in range(self.size)]
        for y in range(self.size):
            check = [(-1, [(y, -1)])] * self.size
            for x in range(self.size):
                digit = copy_matrix[y][x] - 1
                num, coords = check[digit]
                num += 1
                coords.append((y, x))
                check[digit] = (num, coords)
            for index, (num, coords) in enumerate(check):
                if num == 0:
                    continue
                for row, col in coords:
                    if col == -1:
                        col = index
                    optim_matrix[row][col] += 1
        for x in range(self.size):
            check = [(-1, [(-1, x)])] * self.size
            for y in range(self.size):
                digit = copy_matrix[y][x] - 1
                num, coords = check[digit]
                num += 1
                coords.append((y, x))
                check[digit] = (num, coords)
            for index, (num, coords) in enumerate(check):
                if num == 0:
                    continue
                for row, col in coords:
                    if row == -1:
                        row = index
                    optim_matrix[row][col] += 1
        for (gy, gx, ly, lx), c in self.conditions:
            if not self.check_condition(gy, gx, ly, lx, copy_matrix):
                optim_matrix[gy // 2][gx // 2] += 10
                optim_matrix[ly // 2][lx // 2] += 10
        return optim_matrix

    def get_mutations_map(self, solution):
        """
        this function get a solution and returns a list of all the places need to change in order to get better solution
        :param solution: the given solution
        :return: the map of the places
        """
        optim_matrix = self.get_mutations_matrix(solution)
        optim_map = []
        for y in range(self.size):
            for x in range(self.size):
                if self.matrix[y][x] == 0:
                    optim_map.append(optim_matrix[y][x])
        return optim_map


class Solver:
    """
    this class represents our solver algorithm
    """

    def __init__(self, matrix_parameters, board):
        """
        init the solver
        :param matrix_parameters: the parameters of the board
        :param board: the board itself
        """
        self.board = board
        self.k = matrix_parameters["k"]
        self.digits = []
        for d in matrix_parameters["digits"]:
            self.digits.append(d[2])
        # how many mutations we want in each mutat solution
        self.mutations_percentage = DEFAULT_MUTATIONS_PERCENTS
        self.parent_percent = lambda x, min_fitness, max_fitness: (
                ((x - min_fitness) / (max_fitness - min_fitness)) * .5) if max_fitness != min_fitness else 1 / 100
        self.population = 100  # how many solutions in each generation
        self.gen = self.create_random_solutions(self.population)
        self.best_score = -100
        self.worst_score = 100
        self.gen_score = 0
        self.old_best = 0
        self.in_row = 0
        self.best_in_row = 0
        self.local_maximum = False
        self.best_sol = None
        self.old_best_sols = {}
        self.gen_num = 0

    def optimize(self):
        """
        this function is used to optimize our solutions each generation
        :return: new optimize solutions
        """
        optimize_solutions = []
        for solution in self.gen:
            # the indexes of the solution
            indexes = [_ for _ in range(len(solution))]
            # the fitness score before optimization
            solution_before_score = self.board.evaluate_solution(solution)
            s_temp = solution.copy()
            for _ in range(self.k):
                # gets the map of where are all mistakes in the solution (the more problems the grater score)
                mutations_map = self.board.get_mutations_map(s_temp)
                # if there is no need to change anything
                if sum(mutations_map) == 0:
                    break
                i, j = random.choices(indexes, weights = mutations_map, k = 2)
                temp = solution[i]
                s_temp[i] = s_temp[j]
                s_temp[j] = temp
            optim_score = self.board.evaluate_solution(s_temp)
            # if the optimization didn't help reverse it
            if optim_score < solution_before_score:
                s_temp = solution.copy()
            optimize_solutions.append(s_temp)
        return optimize_solutions

    def crossfit(self, parents):
        """
        this function use crossfit to create solution from two given solutions
        :param parents:
        :return:
        """
        s1l, s2l = parents
        s1, fitness_score = s1l
        s2, fitness_score2 = s2l
        # chose index to switch between solutions
        index = random.randint(1, len(s1))
        solution = []
        # chose which solution to start with
        parent_to_take = random.randint(0, 1)
        for i, d in enumerate(zip(s1, s2)):
            if i == index:
                parent_to_take = (parent_to_take + 1) % 2
            solution.append(d[parent_to_take])
        # check solution for to much digits result from the cross fit
        digits_to_change, digits_to_add = self.check_solution(solution)
        for d in digits_to_change:
            i = solution.index(d)
            digit_added = random.choice(digits_to_add)
            solution[i] = digit_added
            digits_to_add.remove(digit_added)
        return solution

    def check_solution(self, s):
        """
        this function checks if the solution has good enough digits
        :param s: the solution
        :return: the number of digits to add and the number of digits remove
        """
        check = [0] * self.k
        for d in self.digits:
            check[d - 1] += 1
        for d in s:
            check[d - 1] += 1
        digits_to_change = []
        digits_to_add = []
        for i, d in enumerate(check):
            if d > self.k:
                for _ in range(d - self.k):
                    digits_to_change.append(i + 1)
            if d < self.k:
                for _ in range(self.k - d):
                    digits_to_add.append(i + 1)
        return digits_to_change, digits_to_add

    def fitness(self, solutions):
        """
        this function get out set solutions and calculate what it's fitness
        :param solutions: the set solutions
        :return: the fitness score list
        """
        fitness_scores = [self.board.evaluate_solution(solution) for solution in solutions]
        best_in_gen = max(fitness_scores)
        self.gen_score = sum(fitness_scores) / len(fitness_scores)
        self.best_score = max(self.best_score, best_in_gen)
        self.worst_score = min(fitness_scores)
        return fitness_scores, (best_in_gen, self.worst_score)

    def new_gen(self, solutions_score, max_score, min_score):
        """
        this function create the new generation solutions
        :param solutions_score: solutions with their fitness score unite
        :param max_score: the max score of this generation
        :param min_score: the lowest score of this generation
        :return: new generation set of solutions
        """
        new_gen = []
        # if the solution score stay the same
        best = str(solutions_score[0])
        if best in self.old_best_sols:
            self.old_best_sols[best][0] += 1
            # check if we are at local maximum
            if self.old_best_sols[best][0] >= DEFAULT_MAXIMUM_LOCAL:
                self.local_maximum = True
        else:
            self.old_best_sols[best] = [1, solutions_score[0]]
        # if we are in local_maximum
        if self.local_maximum:
            new_gen_percent = (0, 0.45, 0.5)
            self.mutations_percentage = DEFAULT_MUTATIONS_PERCENTS * 3
            self.in_row += 1
            if self.in_row > DEFAULT_TIME_RESTARTING:
                self.local_maximum = False
                self.in_row = 0
        else:
            children = random.randint(10, 40)
            mutations = random.randint(10, 40)
            # how many type solutions we gets in the new gen
            new_gen_percent = (0.2, children / 100, mutations / 100)
            self.mutations_percentage = DEFAULT_MUTATIONS_PERCENTS
            self.in_row = 0
        # calculate the chances of each solution to chosen to the crossfit
        chances = [self.parent_percent(fitness_score, min_score, max_score) for solution, fitness_score in
                   solutions_score]
        self.old_best = max_score
        old, children, mutations = new_gen_percent
        children_solutions = 0
        # crossfit
        while children_solutions < children * self.population:
            parents = random.choices(solutions_score, weights = chances, k = 2)
            child = self.crossfit(parents)
            if child not in new_gen:
                new_gen.append(child)
                children_solutions += 1
        # mutations
        mutate_solutions = 0
        while mutate_solutions < mutations * self.population:
            solution, fitness_score = random.choice(solutions_score)
            mutate_solution = self.mutate(solution)
            if mutate_solution not in new_gen:
                new_gen.append(mutate_solution)
                mutate_solutions += 1
        # old solutions
        old_solutions = int(self.population * old)
        old_gen = [solution for solution, fitness_score in solutions_score[:old_solutions]]
        if self.local_maximum:
            half_solutions = self.population // 2
            old_gen = [solution for solution, fitness_score in
                       solutions_score[-old_solutions:]]
        new_gen.extend(old_gen)
        # if we didn't get the exact number of solutions we add random solutions
        if len(new_gen) < self.population:
            new_gen.extend(self.create_random_solutions(self.population - len(new_gen)))
        return new_gen

    def mutate(self, solution):
        """
        this function take a solution and return a mutated solution based on that solution
        :param solution: the solution we want to mutate
        :return: the new solution
        """
        s1 = solution.copy()
        indexes = [j for j in range(len(s1))]
        for _ in range(int(self.mutations_percentage * len(indexes) / 2)):
            if len(indexes) < 2:
                break
            places = random.choices(indexes, k = 2)
            m1, m2 = places
            temp = s1[m1]
            s1[m1] = s1[m2]
            s1[m2] = temp
            m = random.randint(0, 1)
            indexes.remove(places[m])
        return s1

    def create_random_solutions(self, number_of_solutions):
        """
        this function create random solutions given number of solutions we want
        :param number_of_solutions: how many random solutions we need
        :return:
        """
        digit_list = [d for d in range(1, self.k + 1)] * self.k
        for d in self.digits:
            digit_list.remove(d)
        solutions = []
        for _ in range(number_of_solutions):
            s = digit_list.copy()
            random.shuffle(s)
            solutions.append(s)
        return solutions

    def update_solutions(self):
        """
        this function will be implemented in the son classes
        :return:
        """

        pass

    def solve_board(self, screen):
        """
        this function try to solve the board
        :param screen: the screen we use our GUI
        :return:
        """
        bests = []
        worst = []
        scores = []
        best_sols = []
        number_of_generation = 5000
        run = True
        clock = pygame.time.Clock()
        fps = 120
        found_sol = False
        while run:
            clock.tick(fps)
            print(f"Generation : {self.gen_num}")
            self.update_solutions()
            print(f"best Score {self.best_score}")
            print(f"worst Score {self.worst_score}")
            print(f"gen Score {self.gen_score}")
            score = self.board.evaluate_solution(s.best_sol)
            bests.append(self.best_score)
            best_sols.append(score)
            print(f"Best Gen Score : {score}")
            if self.gen_num % 5 == 0:
                scores.append(self.gen_score)
                worst.append(self.worst_score)
            self.gen_num += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return pygame.QUIT
            self.board.draw_solution(screen, self.best_sol, self.gen_num)
            pygame.display.update()
            if self.board.is_solved(self.best_sol):
                run = False
                found_sol = True
            if self.gen_num > number_of_generation:
                run = False
        return found_sol, (bests, worst, scores, best_sols)


class Normal_Solver(Solver):
    def update_solutions(self):
        fitness_scores, (max_score, min_score) = self.fitness(self.gen)
        solutions_with_score = attach_fitness_to_solution(self.gen, fitness_scores)
        self.gen = self.new_gen(solutions_with_score, max_score, min_score)
        self.best_sol = solutions_with_score[0][0]


class Darwin_Solver(Solver):
    def update_solutions(self):
        optimize_solutions = self.optimize()
        fitness_scores, (max_score, min_score) = self.fitness(optimize_solutions)
        solutions_with_score = attach_fitness_to_solution(self.gen, fitness_scores)
        self.gen = self.new_gen(solutions_with_score, max_score, min_score)
        self.best_sol = solutions_with_score[0][0]


class Lamarck_Solver(Solver):
    def update_solutions(self):
        optimize_solutions = self.optimize()
        fitness_scores, (max_score, min_score) = self.fitness(optimize_solutions)
        solutions_with_score = attach_fitness_to_solution(optimize_solutions, fitness_scores)
        self.gen = self.new_gen(solutions_with_score, max_score, min_score)
        self.best_sol = solutions_with_score[0][0]


def attach_fitness_to_solution(solutions, fitness):
    """
    this function attach the fitness score to that solution and sort them
    :param solutions: the solutions
    :param fitness: the fitness scores
    :return: list of sorted solutions with fitness score
    """
    solutions_fitness = []
    for solution, fittness_score in zip(solutions, fitness):
        solutions_fitness.append((solution, fittness_score))
    solutions_fitness.sort(key = lambda x: x[1], reverse = True)
    return solutions_fitness


def plot_generations(best, worst, scores, best_sols):
    plt.plot(best, 'b', label = "best")
    gens = [gen for gen in range(0, len(best), 5)]
    plt.plot(gens, worst, 'r', label = "worst")
    plt.plot(gens, scores, 'y', label = "Gen Score")
    plt.plot(best_sols, 'g', label = "Score of the best solution")
    plt.xlabel("Generation")
    plt.ylabel("score")
    plt.legend(loc = 'lower right')
    plt.show(block = True)


def read_config(config_file):
    matrix_parameters = {}
    with open(config_file, 'r') as file:
        matrix_parameters["k"] = int(file.readline())
        num_digits = int(file.readline())
        digits = []
        for _ in range(num_digits):
            row = file.readline()
            l = []
            for i, char in enumerate(row):
                if i % 2 == 0:
                    l.append(int(char))
            digits.append(l)
        matrix_parameters["digits"] = digits
        num_conditions = int(file.readline())
        conditions = []
        for _ in range(num_conditions):
            row = file.readline()
            l = []
            for i, char in enumerate(row):
                if i % 2 == 0:
                    l.append(int(char))
            conditions.append(l)
        matrix_parameters["conditions"] = conditions
    return matrix_parameters


print("Hello and welcome to Futoshiki Solver")
good_path = False
while not good_path:
    path = input("Please enter the path to the board to solve\n")
    good_path = os.path.exists(path)
    if not good_path:
        print("Please enter a valid path")
solver = -1
solvers = [Normal_Solver, Darwin_Solver, Lamarck_Solver]
while solver < 1 or solver > 3:
    print("Please Choose which solver you want to chose:")
    print("1) Normal Solver")
    print("2) Darwin Solver")
    print("3) Lamarck Solver")
    choice = input("?\n")
    solver = int(choice)
    if solver < 1 or solver > 3:
        print("Please chose a valid choice")
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Futoshiki Solver")
param = read_config(path)
b = Board(param)
s = solvers[solver - 1](param, b)
star_time = pygame.time.get_ticks()
returned_value = s.solve_board(win)
if returned_value == pygame.QUIT:
    pygame.quit()
    quit()
solved, (bests, worst, scores, best_sols) = returned_value
plot_generations(bests, worst, scores, best_sols)
show = True
fps = 60
clock = pygame.time.Clock()
clock.tick(fps)
mutations_map = None
if solved:
    print("Solved!")
    best_sol = s.best_sol
else:
    sort_solutions = sorted(s.old_best_sols.items(), key = lambda x: x[1][1][1])
    best_sol = sort_solutions.pop()[1][1][0]
    mutations_map = b.get_mutations_matrix(best_sol)
print(f"Running Time :{(pygame.time.get_ticks() - star_time) / 1000} seconds")

while show:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            show = False
    b.draw_solution(win, best_sol, s.gen_num, mutations_map)
