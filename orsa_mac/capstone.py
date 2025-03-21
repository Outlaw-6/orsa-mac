#!/usr/bin/env python3

import csv
import math
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

    def __init__(self, location: float):
        self.location = location

    def process_tick(self):
        print("Base Class Tick Process Message")

    def distance(self, other) -> float:
        return(gd.great_circle(self.location, other.location).km)


class Target(SimEntity):

    """A target in the sim.
    """

    def __init__(self, location: tuple[float, float],
                 radius: int, priority: int, name: str):
        self.location = location
        self.area = radius ** 2 * math.pi
        self.priority = priority
        self.radius = radius
        self.name = name


class Weapon:

    """A munition fired from a weapon system
    """

    def __init__(self, range: int, burst_radius: int,
                 reliability: float, jkw_prob: float, cep: int):
        self.range = range
        self.burst_radius = burst_radius
        self.reliability = reliability
        self.jkw_prob = jkw_prob
        self.cep = cep


class WeaponSystem(SimEntity):

    """A weapon system in the sim
    """

    def __init__(self, location: tuple[float, float],
                 weapons: list[Weapon], name: str):
        self.location = location
        self.weapons = weapons
        self.name = name

    # TODO make weighing function
    def weight(self, _target: Target):
        return(5)


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

    m = 10

    n_tgts = len(tgts)
    n_wpns = len(wpn_systems)

    total = 0
    for wpn_sys in wpn_systems:
        for _ in tgts:
            for _ in wpn_sys.weapons:
                total += 1
    total += len(tgts)

    coefficients = [0 for _ in range(total)]

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
            for _, wpn in enumerate(wpn_sys.weapons, start = 1):
                constraints_lhs[i][index] = 1
                if (wpn_sys.distance(tgt) < wpn.range):
                    constraints_lhs[(n_wpns+j)][index] = -1
                    coefficients[index] = wpn_sys.weight(tgt)
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
