#!/usr/bin/env python3

import geopy.distance as gd

"""This is the main file for the Teem Juan ORSA-MAC Capstone Project.

It contains the main logic loop for our ATACMs vs. PRSM simultaion.
"""

__version__ = "0.1"
__author__ = "Luke Miller"

class Sim:
    def __init__(self):
        pass

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
                 area: float, priority: int):
        self.location = location
        self.area = area
        self.priority = priority


class WeaponSystem(SimEntity):

    """A weapon system in the sim
    """

    def __init__(self, location: tuple[float, float],
                 weapons: list):
        self.location = location
        self.weapons = weapons


class Weapon:

    """A munition fired from a weapon system
    """

    def __init__(self, range: float, burst_radius: float,
                 reliability: float, jkw_prob: float, cep: float):
        self.range = range
        self.burst_radius = burst_radius
        self.reliability = reliability
        self.jkw_prob = jkw_prob
        self.cep = cep
