from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.time import RandomActivation
from mesa.time import BaseScheduler
from mesa.space import SingleGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
from random import random, sample

class Environment(Agent):
    """
    A cell representing the Environment that can become ill and recover.
    """

    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.ill = 0  # 0: Healthy,1: Incubated 2: Ill, 3: Recovered, 4: Dead 5. In quarantine
        self.xestelen = 0 # 0: do noting,1: to become incubated 2: to become ill, 3: to recover, 4: to die, 5: to be quarantine
        self.days_with_symptoms = 0
        self.days_in_incubation = 0

    def step(self):
        
        if self.ill == 1:
            # neighbors = self.model.grid.get_neighbors(self.pos, False)
            radius = self.model.radius
            encounters = self.model.encounters
            neighbor_positions = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False, radius=radius)
            potential_infections = []

            for pos in neighbor_positions:
                neighbor_agent = self.model.grid.get_cell_list_contents([pos])[0]
                potential_infections.append(neighbor_agent)

            # Randomly select up to (number of enxounters) agents to infect
            new_infections = sample(potential_infections, min(encounters, len(potential_infections)))
            for n in new_infections:
                if random() < self.model.transmission_rate:
                    if n.ill == 0:
                        n.xestelen = 1

            if self.days_in_incubation == self.model.days_in_incubation:
                if random() < self.model.self_quarantine_rate:
                    self.xestelen = 5
                else: self.xestelen = 2
            else: self.days_in_incubation += 1

        if self.ill == 2:
            if self.days_with_symptoms == self.model.days_with_symptoms:
                if random() < self.model.fatality_rate - self.model.fatality_rate * self.model.hospital_capacity:
                    self.xestelen = 4
                else: self.xestelen = 3
            else: self.days_with_symptoms += 1

        if self.ill == 5:
            radius = self.model.radius
            encounters = self.model.encounters
            neighbor_positions = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False, radius=radius)
            potential_infections = []

            for pos in neighbor_positions:
                neighbor_agent = self.model.grid.get_cell_list_contents([pos])[0]
                potential_infections.append(neighbor_agent)

            # Randomly select up to (number of enxounters) agents to infect
            new_infections = sample(potential_infections, min(encounters, len(potential_infections)))
            for n in new_infections:
                if random() < self.model.self_quarantine_strictness * self.model.transmission_rate:
                    if n.ill == 0:
                        n.xestelen = 1
            
            if self.days_with_symptoms == self.model.days_with_symptoms:
                self.xestelen = 3
            else: self.days_with_symptoms += 1


    def advance(self):
        if self.xestelen == 1:
            self.ill = 1
            self.xestelen = 0
            self.days_in_incubation += 1
        if self.xestelen == 2:
            self.ill = 2
            self.xestelen = 0
            self.days_with_symptoms += 1
        if self.xestelen == 3:
            self.ill = 3
            self.xestelen = 0
        if self.xestelen == 4:
            self.ill = 4
            self.xestelen = 0
        if self.xestelen == 5:
            self.ill = 5
            self.xestelen = 5
            self.days_with_symptoms += 1
        
    def get_neighbors_within_radius(self, radius):
        neighbors = []
        x, y = self.pos
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:  # This ensures it's within the Manhattan distance (radius)
                    n_pos = (x + dx, y + dy)
                    if not self.model.grid.out_of_bounds(n_pos):
                        neighbors.extend(self.model.grid.get_cell_list_contents([n_pos]))
        return neighbors

class Diffusion(Model):
    """
    Represents the diffusion of illness.
    """

    def __init__(self, height=50, width=50, days_with_symptoms=5, days_in_incubation=7, transmission_rate=0.12, radius=8, encounters=4, fatality_rate=0.03, hospital_capacity=0.5, self_quarantine_rate=0.5, self_quarantine_strictness=0.5 ):
        """
        Create a new playing area of (height, width) cells.
        """
        super().__init__()

        self.days_with_symptoms = days_with_symptoms
        self.days_in_incubation = days_in_incubation
        self.transmission_rate = transmission_rate
        self.radius = radius
        self.encounters = encounters
        self.fatality_rate = fatality_rate
        self.hospital_capacity = hospital_capacity
        self.self_quarantine_rate = self_quarantine_rate
        self.self_quarantine_strictness = self_quarantine_strictness

        self.schedule = SimultaneousActivation(self)
        self.grid = SingleGrid(width, height, torus=True)  # Corrected order of dimensions

        self.create_and_place_agents()

        self.running = True

        # Initialize the central cell to be ill
        center_x, center_y = width // 2, height // 2
        center_cell = self.grid.get_cell_list_contents([(center_x, center_y)])[0]
        center_cell.ill = 1
        center_cell.days_in_incubation = 1

    def create_and_place_agents(self):
        """
        Create and place an Environment agent in each cell of the grid.
        """
        for cell_content, (x, y) in self.grid.coord_iter():
            cell = Environment((x, y), self)
            self.grid.place_agent(cell, (x, y))
            self.schedule.add(cell)

    def step(self):
        self.schedule.step()

def diffusion_portrayal(agent):
    if agent.ill == 0:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "white"}
    elif agent.ill == 1:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "pink"}
    elif agent.ill == 2:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "red"}
    elif agent.ill == 3:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "gray"}
    elif agent.ill == 4:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "black"}
    elif agent.ill == 5:
        portrayal = {"Shape": "rect", "w": 1, "h": 1, "Filled": "true", "Layer": 0, "Color": "blue"}

    return portrayal

canvas_element = CanvasGrid(diffusion_portrayal, 50, 50, 500, 500)

model_params = {
    "height": 50,
    "width": 50,
    "days_with_symptoms": Slider("days_with_symptoms", 5, 1, 20, 1),
    "transmission_rate": Slider("transmission_rate", 0.12, 0, 1, 0.01),
    "days_in_incubation": Slider("days_in_incubation", 7, 0, 20, 1),
    "radius": Slider("radius", 8, 0, 25, 1),
    "encounters": Slider("encounters", 4, 1, 30, 1),
    "fatality_rate": Slider("fatality_rate", 0.03, 0, 0.3, 0.01),
    "hospital_capacity": Slider("hospital_capacity", 0.5, 0, 1, 0.01),
    "self_quarantine_rate": Slider("self_quarantine_rate", 0.5, 0, 1, 0.01),
    "self_quarantine_strictness": Slider("self_quarantine_strictness", 0.5, 0, 1, 0.01),
}

server = ModularServer(
    Diffusion, [canvas_element], "Covid", model_params
)

server.port = 8538
server.launch()