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

    Not sure if I actually need it, but it is a convienient place to store the
    distance method.
    """

    def __init__(self, location: tuple[float, float]):
        self.location: tuple[float, float] = location

    def distance(self, other) -> float:
        return(gd.great_circle(self.location, other.location).km)

    highest_cost = 2000000



class Target(SimEntity):

    """A target in the sim.
    """

    def __init__(self,
                 type: str,
                 name: str,
                 location: tuple[float, float],
                 radius: int,
                 priority: int):
        self.location: tuple[float, float] = location
        self.area: float = radius ** 2 * math.pi
        self.priority: int = priority
        self.radius: int = radius
        self.name: str = name
        self.type: str = type


class Weapon:

    """A munition fired from a weapon system.
    """

    def __init__(self,
                 range: int,
                 burst_radius: dict[str,float],
                 reliability: float,
                 jkw_prob: dict[str,float],
                 cep: int,
                 cost: float,
                 type: str):
        self.range:int = range
        self.burst_radius: dict[str,float] = burst_radius
        self.reliability: float = reliability
        self.jkw_prob: dict[str,float] = jkw_prob
        self.cep: int = cep
        self.cost: float = cost
        self.type: str = type

    def burst(self, tgt: Target) -> float:
        return(self.burst_radius[tgt.type])

    def jkw(self, tgt: Target) -> float:
        return(self.jkw_prob[tgt.type])


class WeaponSystem(SimEntity):

    """A weapon system in the sim.
    """

    def __init__(self,
                 location: tuple[float, float],
                 weapons: list[Weapon],
                 name: str,
                 ammo: list[int]):
        self.location: tuple[float, float] = location
        self.weapons: list[Weapon] = weapons
        self.name: str = name
        self.ammo: list[int] = ammo


    def weight(self, tgt: Target, weapon: int) -> float:
        wpn = self.weapons[weapon]
        factor_weights = self.targeting_weights
        cost_factor = wpn.cost/self.highest_cost/factor_weights["cost"]
        jkw_factor = (1-wpn.jkw(tgt))/factor_weights["jkw"]
        reliability_factor = (1-wpn.reliability)/factor_weights["reliability"]
        burst_delta = (wpn.burst(tgt) - tgt.radius)/wpn.burst(tgt)
        if burst_delta > 0:
            cd_factor = burst_delta/factor_weights["cd"]
        else:
            cd_factor = 0

        return(cost_factor + jkw_factor + reliability_factor + cd_factor)


    def range(self, weapon: int) -> float:
        return(self.weapons[weapon].range)

    targeting_weights = {"cost": 1,
                         "cd": 1,
                         "reliability": 1,
                         "jkw": 2}



class F15(WeaponSystem):

    """An F15. They have special rules.
    """

    def __init__(self,
                 location: tuple[float, float],
                 weapons: list[Weapon],
                 name: str,
                 ammo: list[int],
                 take_off_reliability: float,
                 flight_reliability: float,
                 ada_reliability: float,
                 engage_reliability: float,
                 fuel_dist: float):
        self.location: tuple[float, float] = location
        self.weapons: list[Weapon] = weapons
        self.name: str = name
        self.ammo: list[int] = ammo
        self.take_off_reliability: float = take_off_reliability
        self.flight_reliability: float = flight_reliability
        self.ada_reliability: float = ada_reliability
        self.engage_reliability: float = engage_reliability
        self.fuel_dist: float = fuel_dist

    def update_location(self, location: tuple[float, float]):
        distance = self.distance(location)
        if distance > self.fuel_dist:
            print("past fuel range") # TODO put an actual fuel system here
        self.location = location

    def take_off(self) -> bool:
        return(random.random() < self.take_off_reliability)

    def fly(self) -> bool:
        return(random.random() < self.flight_reliability)

    def penetrate_ada(self) -> bool:
        return(random.random() < self.ada_reliability)

    def range(self, weapon: int) -> float:
        wpn = self.weapons[weapon]
        if wpn.type == "GBU10" or wpn.type == "GBU39":
            return(self.fuel_dist)
        else:
            return(self.fuel_dist + wpn.range)

    def weight(self, tgt: Target, weapon: int) -> float:
        # TODO Add logic for going into ADA bubble
        return(super(F15, self).weight(tgt, weapon))


def scenario_1():

    """Main Loop for Scenario One.
    """

    # Load Targets TODO redo
    targets: list[Target] = []
    with open("./target_lat_lon.csv", mode="r") as f:
        csv_tgts = list(csv.reader(f))
        for line in csv_tgts[1:]:
            name = line[1]
            lat = float(line[2])
            lon = float(line[3])
            targets.append(Target("CT","CT1",(lat,lon),0,0))

    # Load MLRS TODO redo
    mlrs: list[WeaponSystem] = []
    with open("./MLRS_BN_lat_lon.csv", mode="r") as f:
        csv_mlrs = list(csv.reader(f))
        for line in csv_mlrs[1:]:
            name = line[1]
            lat = float(line[2])
            lon = float(line[3])
            mlrs += [WeaponSystem((lat,lon),[],name,[1])]

    # Load SAMS
    sams: list[Target] = []
    sam_radars: list[Target] = []
    with open("./sams.csv", mode="r") as f:
        csv_sams = list(csv.reader(f))
        for i, line in enumerate(csv_sams[1:]):
            type,name,lat,lon,radius,priority = line
            lat = float(lat)
            lon = float(lon)
            radius = int(radius)
            priority = int(priority)
            sam_radars.append(Target(type,name,(lat,lon),radius,priority))

            # Add a SAM site within 50km of the radar at random
            new_lat = lat - 0.5 + random.random()
            new_lon = lon - 0.5 + random.random()
            while (gd.great_circle((lat,lon),new_lat,new_lon).km > 50):
                new_lat = lat - 0.5 + random.random()
                new_lon = lon - 0.5 + random.random()
            sams.append(Target("SAM","SAM " + str(i),
                               (new_lat, new_lon), radius, priority))

    # Load Weapon Types
    weapons: dict[str,Weapon] = {}
    with open("./wpn_stats.csv", mode="r") as f:
        csv_wpns = list(csv.reader(f))
        current_wpn = csv_wpns[1][0]
        wpn = current_wpn
        range = 0
        burst: dict[str,float] = {}
        reliability = 0
        jkw: dict[str,float] = {}
        cep = 0
        cost = 0
        for line in csv_wpns[1:]:
            wpn = line[0]
            if wpn != current_wpn:
                weapons[current_wpn] = Weapon(range,burst,reliability,
                                              jkw,cep,cost,current_wpn)
                burst = {}
                jkw = {}
            wpn, tgt, burst_r, jkw_p, reliability, range, cep, cost = line
            reliability = float(reliability)
            cep = int(cep)
            cost = int(cost)
            range = int(range)
            jkw[tgt] = float(jkw_p)
            burst[tgt] = float(burst_r)
        weapons[wpn] = Weapon(range,burst,reliability,jkw,cep,cost,wpn)


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
            for k, _ in enumerate(wpn_sys.weapons):
                constraints_lhs[i][index] = 1
                if (wpn_sys.distance(tgt) < wpn_sys.range(k)):
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
