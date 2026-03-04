import mesa
from mesa.time import RandomActivation
from mesa import Model
from agent_student import Student
import random

class CampusModel(Model):
    def __init__(self, G_all, G_drive, nodes_proj, doors, n_students):
        super().__init__()
        self.G_all = G_all
        self.G_drive = G_drive
        self.nodes_proj = nodes_proj
        self.building_doors = doors
        self.schedule = mesa.time.RandomActivation(self)
        self.n_students = n_students

        all_nodes_ids = list(self.nodes_proj.index)
        for i in range(n_students):
            start_node = random.choice(all_nodes_ids)
            delay = 0
            a = Student(i, self, start_node, delay=delay, indoors=False)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()