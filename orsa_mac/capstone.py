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
    def weight(self):
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


def targeting(mlrs: list[WeaponSystem], tgts: list[Target]):

    # Get total length of our array of DVs
    total = (len(mlrs) + 1) * len(tgts)
    coefficients = [0 for _ in range(total)]
    dvs = [0 for _ in range(total)]

    m = 1000000

    n_tgts = len(tgts)
    n_mlrs = len(mlrs)

    # Make the array of LHS coefficents and vector of RHS values
    constraints_lhs = [[0 for _ in range(total)] for _ in range(n_mlrs + n_tgts + 1)]
    constraints_rhs = [0 for _ in range(n_mlrs + n_tgts + 1)]

    # Go through all weapon/target combinations
    for i, bn in enumerate(mlrs, start = 1):
        # Add weapon constraint RHS
        constraints_rhs[i-1] = 1

        # Build constraint vector for combination
        for j, tgt in enumerate(tgts, start = 1):
            index = i * n_tgts - (n_tgts - (j-1))
            constraints_lhs[(i-1)][index] = 1
            if (bn.distance(tgt) < 300):
                constraints_lhs[n_mlrs+j-1][index] = (-1) * bn.weapons[0].burst_radius
                coefficients[index] = bn.weight()
            else:
                coefficients[index] = 2*m

    # Add target constraint RHS
    for j, tgt in enumerate(tgts):
        constraints_rhs[j+len(mlrs)] = tgt.radius * (-1)

    # Add dummy node
    constraints_rhs[-1] = m
    for i in range(n_tgts):
        index = (n_mlrs + 1) * n_tgts - (n_tgts - i)
        constraints_lhs[n_mlrs+i][index] = m
        #TODO put priority checking in here. High priority node cost for dummy is m
        coefficients[index] = 20
