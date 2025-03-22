#!/usr/bin/env python3

import csv
import math
import random
import geopy.distance as gd
import scipy.optimize as opt

"""This is the main file for the Teem Juan ORSA-MAC Capstone Project.

It contains the main logic loop for our ATACMs vs. PRSM simultaion.
"""

__version__ = "0.1"
__author__ = "Luke Miller"


class SimEntity:

    """A superclass to both targets and weapon systems. Has a location.
    """

    def __init__(self, location: tuple[float, float]):
        self.location: tuple[float, float] = location

    def process_tick(self):
        print("Base Class Tick Process Message")

    def distance(self, other) -> float:
        return(gd.great_circle(self.location, other.location).km)

    highest_cost = 2000000

    targeting_weights = {"cost": 1,
                         "cd": 1,
                         "reliability": 1,
                         "jkw": 2}


class Target(SimEntity):

    """A target in the sim.
    """

    def __init__(self, location: tuple[float, float],
                 radius: int, priority: int, name: str):
        self.location: tuple[float, float] = location
        self.area: float = radius ** 2 * math.pi
        self.priority: int = priority
        self.radius: int = radius
        self.name: str = name


class Weapon:

    """A munition fired from a weapon system
    """

    def __init__(self,
                 range: int,
                 burst_radius: int,
                 reliability: float,
                 jkw_prob: float,
                 cep: int,
                 cost: float):
        self.range:int = range
        self.burst_radius: int = burst_radius
        self.burst_area: float = burst_radius ** 2 * math.pi
        self.reliability: float = reliability
        self.jkw_prob: float = jkw_prob
        self.cep: int = cep
        self.cost: float = cost


class WeaponSystem(SimEntity):

    """A weapon system in the sim
    """

    def __init__(self,
                 location: tuple[float, float],
                 weapons: list[Weapon],
                 name: str):
        self.location = location
        self.weapons = weapons
        self.name = name


    def weight(self, tgt: Target, weapon: int) -> float:
        wpn = self.weapons[weapon]
        factor_weights = self.targeting_weights
        cost_factor = wpn.cost/self.highest_cost/factor_weights["cost"]
        jkw_factor = (1-wpn.jkw_prob)/factor_weights["jkw"]
        reliability_factor = (1-wpn.reliability)/factor_weights["reliability"]
        burst_delta = (wpn.burst_radius - tgt.radius)/wpn.burst_radius
        if burst_delta > 0:
            cd_factor = burst_delta/factor_weights["cd"]
        else:
            cd_factor = 0

        return(cost_factor + jkw_factor + reliability_factor + cd_factor)


class F15(WeaponSystem):

    """An F15. They have special rules.
    """

    def __init__(self,
                 location: tuple[float, float],
                 weapons: list[Weapon],
                 name: str,
                 take_off_reliability: float,
                 flight_reliability: float,
                 ada_reliability: float,
                 engage_reliability: float):
        self.location: tuple[float, float] = location
        self.weapons: list[Weapon] = weapons
        self.name: str = name
        self.take_off_reliability: float = take_off_reliability
        self.flight_reliability: float = flight_reliability
        self.ada_reliability: float = ada_reliability
        self.engage_reliability: float = engage_reliability

    def update_location(self, location: tuple[float, float]):
        self.location = location

    def take_off(self) -> bool:
        return(random.random() < self.take_off_reliability)

    def fly(self) -> bool:
        return(random.random() < self.flight_reliability)

    def penetrate_ada(self) -> bool:
        return(random.random() < self.ada_reliability)


def main():

    """Main Loop.
    """

    # Load Targets
    targets: list[Target] = []
    with open("./target_lat_lon.csv", mode="r") as f:
        csv_tgts = list(csv.reader(f))
        for line in csv_tgts[1:]:
            name = line[1]
            lat = float(line[2])
            lon = float(line[3])
            targets += [Target((lat,lon),0,0,name)]

    #Load MLRS
    mlrs: list[WeaponSystem] = []
    with open("./MLRS_BN_lat_lon.csv", mode="r") as f:
        csv_mlrs = list(csv.reader(f))
        for line in csv_mlrs[1:]:
            name = line[1]
            lat = float(line[2])
            lon = float(line[3])
            mlrs += [WeaponSystem((lat,lon),[],name)]


def targeting(wpn_systems: list[WeaponSystem], tgts: list[Target]):

    """Builds a linear program to solve for optimal weapon assignment.

    Returns an array in the order of weapon system(i) - target(j) - weapon(k).

    A one in the return array corresponds to engaging with the weapon at ijk
    position when iterating through all i weapon systems, j targets and k
    weapons.

    """

    m = 10

    n_tgts = len(tgts)
    n_wpns = len(wpn_systems)

    total = 0
    for wpn_sys in wpn_systems:
        for _ in tgts:
            for _ in wpn_sys.weapons:
                total += 1
    total += len(tgts)

    coefficients = [0.0 for _ in range(total)]

    # Make the array of LHS coefficents and vector of RHS values
    constraints_lhs = [[0 for _ in range(total)] for _ in range(n_wpns + n_tgts)]
    constraints_rhs = [0 for _ in range(n_wpns + n_tgts)]

    index = 0

    # Go through all weapon/target combinations
    for i, wpn_sys in enumerate(wpn_systems):
        # Add weapon constraint RHS
        constraints_rhs[i] = 1

        # Build constraint vector for combination
        for j, tgt in enumerate(tgts):
            for k, wpn in enumerate(wpn_sys.weapons):
                constraints_lhs[i][index] = 1
                if (wpn_sys.distance(tgt) < wpn.range):
                    constraints_lhs[(n_wpns+j)][index] = -1
                    coefficients[index] = wpn_sys.weight(tgt, k)
                else:
                    constraints_lhs[(n_wpns+j)][index] = 0
                    coefficients[index] = 2*m
                index += 1

    # Add target constraint RHS
    for j, tgt in enumerate(tgts):
        constraints_rhs[j+n_wpns] = -1

    # Add dummy node
    for j, tgt in enumerate(tgts):
        # Add dummy node coefficients
        constraints_lhs[(n_wpns+j)][index] = -m

        # Check priority. High priority doesn't get dummy node.
        if (tgt.priority >= 1):
            coefficients[index] = m
        else:
            coefficients[index] = 1
        index += 1

    # TODO remove these debugging prints
    print(f"LHS: {constraints_lhs}")
    print(f"RHS: {constraints_rhs}")
    print(f"c: {coefficients}")

    # Send to the Solver
    res = opt.linprog(c=coefficients, A_ub=constraints_lhs, b_ub=constraints_rhs,
                      bounds=(0,1), integrality=1)

    return(res.x)
