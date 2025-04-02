#!/usr/bin/env python3

import csv
from datetime import datetime
import math
import random
import multiprocessing
import os
from typing import Sequence
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

    def __eq__(self, other):
        if isinstance(other, Target):
            return(self.name == other.name)
        else:
            return(False)

    def __str__(self):
        return(self.name)

    def __repr__(self):
        return(self.name)


class TimeSensitiveTarget(Target):

    """A Time Sensitive target in the sim. Used in Scenario 2.

    """

    def __init__(self,
                 type: str,
                 name: str,
                 location: tuple[float, float],
                 radius: int,
                 priority: int,
                 moving: bool,
                 density: int):
        self.location: tuple[float, float] = location
        self.area: float = radius ** 2 * math.pi
        self.priority: int = priority
        self.radius: int = radius
        self.name: str = name
        self.type: str = type
        self.moving: bool = moving
        self.density: int = density

    def stop(self) -> bool:
        self.moving = random.random() < 0.5
        return(self.moving)

    turns_seen: int = 0

    def update_seen(self) -> None:
        self.turns_seen += 1

    def target_down(self) -> None:
        self.turns_seen = 0

    def is_moving(self) -> bool:
        return(self.moving)



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

    def __eq__(self, other):
        if isinstance(other, Weapon):
            return self.type == other.type
        else:
            return(False)

    def __str__(self):
        return(self.type)

    def __repr__(self):
        return(self.type)

    def burst(self, tgt: Target) -> float:
        return(self.burst_radius[tgt.type])

    def area(self, tgt: Target) -> float:
        burst_r = self.burst_radius[tgt.type]
        return(burst_r ** 2 * math.pi)

    def jkw(self, tgt: Target) -> float:
        return(self.jkw_prob[tgt.type])


class WeaponSystem(SimEntity):

    """A weapon system in the sim.

    """

    def __init__(self,
                 location: tuple[float, float],
                 weapons: list[Weapon],
                 name: str,
                 ammo: list[int],
                 time_to_reload: int):
        self.location: tuple[float, float] = location
        self.weapons: list[Weapon] = weapons
        self.name: str = name
        self.ammo: list[int] = ammo
        self.time_to_reload: int = time_to_reload
        self.rounds_until_reload: int = time_to_reload


    def __eq__(self, other):
        if isinstance(other, WeaponSystem):
            return(self.name == other.name)
        else:
            return(False)


    def __str__(self):
        return(self.name)

    def __repr__(self):
        return(self.name)

    def weight(self, tgt: Target, weapon: int) -> float:
        wpn = self.weapons[weapon]
        #cost_factor = wpn.cost/self.highest_cost
        jkw_factor = (1-wpn.jkw(tgt))
        reliability_factor = (1-wpn.reliability)
        burst_delta = (wpn.area(tgt) - tgt.area)/wpn.area(tgt)
        if burst_delta > 0:
            cd_factor = burst_delta
        else:
            cd_factor = 0

        return(jkw_factor + reliability_factor + cd_factor)
        #return(cost_factor + jkw_factor + reliability_factor + cd_factor)


    def range(self, weapon: int) -> float:
        return(self.weapons[weapon].range)


    def dud(self, weapon: int) -> bool:
        wpn = self.weapons[weapon]
        if random.random() < wpn.reliability:
            return(False)
        else:
            self.ammo[weapon] -= 1
            if self.ammo[weapon] < 1:
                del(self.ammo[weapon])
                del(self.weapons[weapon])
            self.rounds_until_reload =- 1
            return(True)


    def engage(self, weapon: int, target: Target) -> bool:

        """Engages target with selected weapon.

        Decrements ammo for weapon used, and removes them from the list once
        they are expended.

        """

        wpn = self.weapons[weapon]
        self.ammo[weapon] -= 1
        self.rounds_until_reload -= 1
        if self.ammo[weapon] < 1:
            del(self.ammo[weapon])
            del(self.weapons[weapon])
        return(random.random() < wpn.jkw(target))


    def ready(self) -> bool:
        return(self.rounds_until_reload > 0)


    def reload(self):
        self.rounds_until_reload = self.time_to_reload


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
                 fuel_dist: float,
                 sams: list[Target],
                 time_to_reload: int):
        self.location: tuple[float, float] = location
        self.original_location = location
        self.weapons: list[Weapon] = weapons
        self.name: str = name
        self.ammo: list[int] = ammo
        self.take_off_reliability: float = take_off_reliability
        self.flight_reliability: float = flight_reliability
        self.ada_reliability: float = ada_reliability
        self.engage_reliability: float = engage_reliability
        self.fuel_dist: float = fuel_dist
        self.sams: list[Target] = sams
        self.time_to_reload: int = time_to_reload
        self.rounds_until_reload: int = time_to_reload

    def update_location(self, location: tuple[float, float]):
        distance = self.distance(SimEntity(location))
        if distance > self.fuel_dist:
            print("past fuel range") # TODO put an actual fuel system here
            self.fuel_dist -= distance
            self.location = location

    def _jassm_fly_loc(self, target: Target, range: float) -> tuple[float, float]:
        x1, y1 = target.location
        d = self.distance(target)
        t = range/d
        xt = (1-t)*x1 + t*x1
        yt = (1-t)*y1 + t*y1
        return((xt,yt))

    def _inside_ada(self, obj: SimEntity) -> bool:
        for sam in self.sams:
            if sam.distance(obj) < 250:
                return(True)
        return(False)

    def take_off(self) -> bool:
        return(random.random() < self.take_off_reliability)

    def fly(self) -> bool:
        return(random.random() < self.flight_reliability)

    def penetrate_ada(self, target: Target, weapon: int) -> bool:
        wpn: Weapon = self.weapons[weapon]

        bomb: bool = wpn.type == "GBU10" or wpn.type == "GBU39"
        now_in_ada = self._inside_ada(self)
        tgt_in_ada = self._inside_ada(target)

        if (bomb and tgt_in_ada) or now_in_ada:
            return(random.random() < self.ada_reliability)
        else:
            return(True)

    def after_engage_fly(self) -> bool:
        return(random.random() < self.engage_reliability)

    def range(self, weapon: int) -> float:
        wpn = self.weapons[weapon]
        if wpn.type == "GBU10" or wpn.type == "GBU39":
            return(self.fuel_dist)
        else:
            return(self.fuel_dist + wpn.range)

    def weight(self, tgt: Target, weapon: int) -> float:
        ada = 0
        if self._inside_ada(tgt):
            ada = 0.5
        return(super(F15, self).weight(tgt, weapon) + ada)

    def engage(self, weapon: int, target: Target) -> bool:
        wpn = self.weapons[weapon]

        # Update locations
        if wpn.type == "GBU39" or wpn.type == "GBU10":
            self.update_location(target.location)
        elif self.distance(target) < wpn.range:
            self.update_location(self._jassm_fly_loc(target, wpn.range))

        if wpn.type == "GBU39":
            return(super(F15, self).engage(weapon, target))
        else:
            for i, w in enumerate(self.weapons):
                if w.type != "GBU10":
                    self.ammo[i] -= 1
                if self.ammo[i] == 0:
                    del(self.ammo[i])
                    del(self.weapons[i])
            return(random.random() < wpn.jkw(target))

    def empty(self) -> bool:
        return(self.fuel_dist < 125)

    def refuel(self):
        self.update_location(self.original_location)
        self.rounds_until_reload = self.time_to_reload
        self.fuel_dist = 1250


    # Load Weapon Types
def load_weapons(weapon_csv: str) -> dict[str,Weapon]:
    weapons: dict[str,Weapon] = {}
    with open(weapon_csv, mode="r") as f:
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
                current_wpn = wpn
            wpn, tgt, burst_r, jkw_p, reliability, range, cep, cost = line
            reliability = float(reliability)
            cep = int(cep)
            cost = int(cost)
            range = int(range)
            jkw[tgt] = float(jkw_p)
            burst[tgt] = float(burst_r)
        weapons[wpn] = Weapon(range,burst,reliability,jkw,cep,cost,wpn)
    return(weapons)

    # Load MLRS
def load_mlrs(army_csv: str, weapons: dict[str,Weapon]) -> list[WeaponSystem]:
    mlrs: list[WeaponSystem] = []
    with open(army_csv, mode="r") as f:
        csv_mlrs = list(csv.reader(f))
        for line in csv_mlrs[1:]:
            _type,name,lat,lon,ammo,wpn,ttr = line
            mlrs.append(WeaponSystem((float(lat),float(lon)),
                                     [weapons[wpn]],name,[int(ammo)],
                                     int(ttr)))
    return(mlrs)

    # Load DDG
def load_ddg(navy_csv: str, weapons: dict[str,Weapon]) -> list[WeaponSystem]:
    ddg: list[WeaponSystem] = []
    with open(navy_csv, mode="r") as f:
        csv_ddg = list(csv.reader(f))
        t2,t3,slam,slamer,ttr = csv_ddg[0][4:]
        t2 = weapons[t2]
        t3 = weapons[t3]
        slam = weapons[slam]
        slamer = weapons[slamer]
        for line in csv_ddg[1:]:
            _type,name,lat,lon,t2_a,t3_a,slam_a,slamer_a,ttr = line
            ddg.append(WeaponSystem((float(lat),float(lon)),
                                    [t2,t3,slam,slamer],
                                    name,
                                    [int(t2_a),int(t3_a),
                                     int(slam_a),int(slamer_a)],
                                    int(ttr)))
    return(ddg)

    # Load Targets
def load_targets(target_csv: str) -> list[Target]:
    targets: list[Target] = []
    with open(target_csv, mode="r") as f:
        csv_tgts = list(csv.reader(f))
        for line in csv_tgts[1:]:
            type,name,lat,lon,radius,priority = line
            targets.append(Target(type,name,(float(lat),float(lon)),
                                  int(radius),int(priority)))
    return(targets)

def load_tst(target_csv: str) -> list[TimeSensitiveTarget]:

    """Loads the time-sensitive targets into the sim.

    Their locations are randomly generated along the equator
    with 80 within 300km of 0,0 and 20 between 300-500km.

    """

    def lon_far() -> float:
        return(2.6979 + random.random() * 1.7986)

    def lon_near() -> float:
        return(random.random() * 2.6979)

    targets: list[TimeSensitiveTarget] = []
    near, far = 0, 0
    with open(target_csv, mode="r") as f:
        csv_tgts = list(csv.reader(f))
        for line in csv_tgts[1:]:
            type,name,radius,priority,moving,density = line
            lat: float = 0
            near_full = near >= 80
            far_full = far >= 20

            if random.random() < 0.8:
                if near_full:
                    lon = lon_far()
                    far += 1
                else:
                    lon = lon_near()
                    near += 1
            else:
                if far_full:
                    lon = lon_near()
                    near += 1
                else:
                    lon = lon_far()
                    far += 1
            targets.append(TimeSensitiveTarget(type,
                                               name,
                                               (lat,lon),
                                               int(radius),
                                               int(priority),
                                               moving == "TRUE",
                                               int(density)))
    return(targets)

    # Load SAMS
def load_sams(sam_csv: str) -> list[list[Target]]:
    sams: list[Target] = []
    sam_radars: list[Target] = []
    with open(sam_csv, mode="r") as f:
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
            while (gd.great_circle((lat,lon),(new_lat,new_lon)).km > 50):
                new_lat = lat - 0.5 + random.random()
                new_lon = lon - 0.5 + random.random()
            sams.append(Target("SAM","SAM " + str(i+1),
                               (new_lat, new_lon), radius, priority))
    return([sams,sam_radars])

    # Load F-15s
def load_f_15s(usaf_csv: str,
               weapons: dict[str,Weapon],
               sams: list[Target]) -> list[F15]:
    f_15s: list[F15] = []
    with open(usaf_csv, mode="r") as f:
        csv_usaf = list(csv.reader(f))
        GBU39,GBU10,JASSM,JASSMER = csv_usaf[0][4:8]
        GBU39 = weapons[GBU39]
        GBU10 = weapons[GBU10]
        JASSM = weapons[JASSM]
        JASSMER = weapons[JASSMER]
        for line in csv_usaf[1:]:
            _type,name,lat,lon,g_39_a,g_10_a,j_a,j_er_a,t_o,fly,ada,engage,ttr= line
            if int(j_er_a) == 0:
                wpn = [GBU39,GBU10,JASSM]
                ammo = [int(g_39_a), int(g_10_a), int(j_a)]
            else:
                wpn = [GBU39,GBU10,JASSM,JASSMER]
                ammo = [int(g_39_a), int(g_10_a), int(j_a), int(j_er_a)]
            f_15s.append(F15((float(lat),float(lon)),
                             wpn, name, ammo,
                             float(t_o),float(fly),float(ada),float(engage),
                             1250, sams, int(ttr)))

    return(f_15s)


def targeting(wpn_systems: Sequence[WeaponSystem],
              tgts: Sequence[Target]) -> list[tuple[WeaponSystem, Target, int]]:

    """Builds a linear program to solve for optimal weapon assignment.

    Returns an array in the order of weapon system(i) - target(j) - weapon(k).

    A one in the return array corresponds to engaging with the weapon at ijk
    position when iterating through all i weapon systems, j targets and k
    weapons.

    """

    m = 20

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
                    coefficients[index] = 5*m
                index += 1

    # Add target constraint RHS
    for j, tgt in enumerate(tgts):
        constraints_rhs[j+n_wpns] = -1

    # Add dummy node
    for j, tgt in enumerate(tgts):
        # Add dummy node coefficients
        constraints_lhs[(n_wpns+j)][index] = -m

        # Check priority. High priority doesn't get dummy node.
        coefficients[index] = 2 + tgt.priority * m
        index += 1

    # Send to the Solver
    res = opt.linprog(c=coefficients, A_ub=constraints_lhs, b_ub=constraints_rhs,
                      bounds=(0,1), integrality=1)

    results = []
    index = 0
    for i, wpn_sys in enumerate(wpn_systems):
        for j, tgt in enumerate(tgts):
            for k in range(len(wpn_sys.weapons)):
                if res.x[index] == 1:
                    results.append((wpn_sys, tgt, k))
                index += 1

    return(results)


def weapon_effects(pairings: list[tuple[WeaponSystem, Target, int]]) -> list:
    destroyed: list[Target] = []
    hits: list[tuple[WeaponSystem, Target, Weapon]] = []
    miss: list[tuple[WeaponSystem, Target, Weapon]] = []
    duds: list[tuple[WeaponSystem, Weapon]] = []
    f_15_down: list[tuple[F15, Target]] = []

    record_format = ["wpn sys", "tgt", "wpn", "dud", "hit",
                     "f15_fly", "f15_ada", "f15_engage"]

    turn_record = []

    for pair in pairings:
        wpn_sys, tgt, wpn_index = pair
        wpn = wpn_sys.weapons[wpn_index]

        record = [wpn_sys, tgt, wpn] + [None for _ in range(5)]
        if isinstance(wpn_sys, F15):
            # Do F-15 things. Break if destroyed.
            if not wpn_sys.fly():
                f_15_down.append((wpn_sys, tgt))
                record[record_format.index("f15_fly")] = False
                break
            else:
                record[record_format.index("f15_fly")] = True
            if not wpn_sys.penetrate_ada(tgt, wpn_index):
                f_15_down.append((wpn_sys, tgt))
                record[record_format.index("f15_ada")] = False
                break
            else:
                record[record_format.index("f15_ada")] = True
        if wpn_sys.dud(wpn_index):
            duds.append((wpn_sys, wpn))
            record[record_format.index("dud")] = True
        else:
            record[record_format.index("dud")] = False
            if wpn_sys.engage(wpn_index, tgt):
                # destroy target
                hits.append((wpn_sys, tgt, wpn))
                destroyed.append(tgt)
                record[record_format.index("hit")] = True
            else:
                # mark a miss
                miss.append((wpn_sys, tgt, wpn))
                record[record_format.index("hit")] = False
        if isinstance(wpn_sys, F15):
            if not wpn_sys.after_engage_fly():
                f_15_down.append((wpn_sys, tgt))
                record[record_format.index("f15_engage")] = False
            else:
                record[record_format.index("f15_engage")] = True
        turn_record.append(record)

    return([hits,miss,duds,destroyed,f_15_down,turn_record])


def main(scenario: int = 1, weapon: str = "ATACM", n: int = 10) -> None:

    if not (weapon == "ATACM" or weapon == "PRSM1" or weapon == "PRSM2"):
        print("Scenario must be 1 or 2, and Weapon must be ATACM, PRSM1 or PRSM2")
        return(None)

    lock = multiprocessing.Lock()

    scenario_function: function

    match scenario:
        case 1:
            scenario_function = scenario_1
        case 2:
            scenario_function = scenario_2
        case _:
            print("Scenario must be 1 or 2, and Weapon must be ATACM, PRSM1 or PRSM2")
            return(None)

    file = ("./output/scenario_"+str(scenario)+"_"+str(n)+"_runs "+weapon
            +" "+str(datetime.now())+".csv")
    record_format = ["run", "turn", "wpn sys", "tgt", "wpn", "dud",
                     "hit", "f15_fly", "f15_ada", "f15_engage",
                     "cost", "cd"]
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(record_format)

    print("Starting Main Loop")
    start_time = datetime.now()

    processes = []

    random.seed(123)

    iteration = 1
    processors = os.process_cpu_count()

    if processors:
        while n > processors:
            for _ in range(processors):
                p = multiprocessing.Process(target = scenario_function, args = (weapon, file, lock, iteration))
                processes.append(p)
                p.start()
                iteration += 1
            for p in processes:
                p.join()
            n -= processors
            processes = []

        for _ in range(n):
            p = multiprocessing.Process(target = scenario_function, args = (weapon, file, lock, iteration))
            processes.append(p)
            p.start()
            iteration += 1
        for p in processes:
            p.join()
    else:
        for i in range(n):
            scenario_function(weapon,file,lock,i)
    print("Done. Total time: ", str(datetime.now()-start_time))


def scenario_1(mlrs_type: str, filename: str, lock, run: int = 1) -> None:

    """ Execute scenario one. This is the main loop with all bookkeeping.

    """

    # Load all entities
    weapons: dict[str, Weapon] = load_weapons("./wpn_stats.csv")
    mlrs: list[WeaponSystem] = load_mlrs("./"+mlrs_type+".csv", weapons)
    ddg: list[WeaponSystem] = load_ddg("./ddg.csv", weapons)
    targets: list[Target] = load_targets("./target_lat_lon.csv")
    sams, sam_radars = load_sams("./sams.csv")
    f_15s: list[F15] = load_f_15s("./usaf.csv", weapons, sams)

    f_15s.reverse()

    destroyed_f_15s: list[F15] = []

    f_15s_on_sortie: list[F15] = []

    undestroyed_targets = targets + sams + sam_radars
    destroyed_targets = []
    remaining_sams = sams

    turn = 0

    start_time = datetime.now()

    output = []

    while len(undestroyed_targets):

        print("Run ", run, " Turn ", turn + 1)
        print("Targets Remaining: {left}".format(left = len(undestroyed_targets)))

        if turn%6 == 0:
            print("Update F-15s")
            f_15s = f_15s_on_sortie + f_15s
            f_15s_on_sortie = []
            for f15 in f_15s:
                f15.refuel()
            while len(f_15s_on_sortie) < 9:
                if f_15s[-1].take_off():
                    f_15s_on_sortie.append(f_15s.pop())
                else:
                    f_15s = [f_15s.pop()] + f_15s
            print("F-15s on sortie: ", f_15s_on_sortie)

        turn += 1
        ready_wpn_sys: list[WeaponSystem] = []

        for wpn_sys in mlrs + ddg + f_15s_on_sortie:
            if wpn_sys.ready():
                ready_wpn_sys.append(wpn_sys)
            else:
                wpn_sys.reload()

        target_pairings = targeting(ready_wpn_sys, undestroyed_targets)

        _hits, _miss, _duds, destroyed, f_15_down, record = weapon_effects(target_pairings)

        destroyed_targets = destroyed_targets + destroyed

        print("destroyed targets this round:")
        print(destroyed)
        for target in destroyed:
            del(undestroyed_targets[undestroyed_targets.index(target)])

        # Determine remaining SAM sites
        remaining_sams = []
        remaining_sam_radars = []
        for target in undestroyed_targets:
            if target.type == "SAM":
                remaining_sams.append(target)
            if target.type == "SAMR":
                remaining_sam_radars.append(target)

        # Remove F-15s that were shot down
        destroyed_f_15s = destroyed_f_15s + f_15_down
        for f15 in f_15_down:
            del(f_15s_on_sortie[f_15s_on_sortie.index(f15[0])])

        # Have any F-15s out of fuel land to refuel
        f15_active = f_15s_on_sortie[:]
        for f15 in f15_active:
            if f15.empty():
                f15.refuel()
                f_15s = [f15] + f_15s
                del(f_15s_on_sortie[f_15s_on_sortie.index(f15)])

        # Update the list of SAMS F-15s are checking against
        for f15 in f_15s + f_15s_on_sortie:
            f15.sams = remaining_sams

        # Build output that will be written as csv file
        output.append(stats(record, run, turn))

    print("Runtime: ",str(datetime.now()-start_time))
    lock.acquire()
    try:
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerows(output)
    finally:
        lock.release()

def scenario_2(mlrs_type: str, filename: str, lock, run: int = 1) -> None:

    """ Execute scenario two. This is the main loop with all bookkeeping.

    """

    # Load all entities
    weapons: dict[str, Weapon] = load_weapons("./wpn_stats.csv")
    mlrs: list[WeaponSystem] = load_mlrs("./"+mlrs_type+"_2.csv", weapons)
    ddg: list[WeaponSystem] = load_ddg("./ddg_2.csv", weapons)
    targets: list[TimeSensitiveTarget] = load_tst("./tst.csv")
    f_15s: list[F15] = load_f_15s("./usaf_2.csv", weapons, [])

    f_15s.reverse()

    destroyed_f_15s: list[F15] = []

    f_15s_on_sortie: list[F15] = []

    undestroyed_targets = targets
    destroyed_targets = []

    # TODO Make initial list of available targets

    turn = 0

    output = []

    start_time = datetime.now()

    while len(undestroyed_targets):

        print("Run ", run, " Turn ", turn + 1)
        print("Targets Remaining: {left}".format(left = len(undestroyed_targets)))

        if turn%6 == 0:
            print("Update F-15s")
            f_15s = f_15s_on_sortie + f_15s
            f_15s_on_sortie = []
            for f15 in f_15s:
                f15.refuel()
            while len(f_15s_on_sortie) < 3:
                if f_15s[-1].take_off():
                    f_15s_on_sortie.append(f_15s.pop())
                else:
                    f_15s = [f_15s.pop()] + f_15s
            print("F-15s on sortie: ", f_15s_on_sortie)

        turn += 1
        ready_wpn_sys: list[WeaponSystem] = []

        for wpn_sys in mlrs + ddg + f_15s_on_sortie:
            if wpn_sys.ready():
                ready_wpn_sys.append(wpn_sys)
            else:
                wpn_sys.reload()

        # TODO Build list of targets that appear each round.
        # TODO Check if targets are moving. Visible non-moving get added to targeting

        target_pairings = targeting(ready_wpn_sys, undestroyed_targets)

        _hits, _miss, _duds, destroyed, f_15_down, record = weapon_effects(target_pairings)

        destroyed_targets = destroyed_targets + destroyed

        print("destroyed targets this round:")
        print(destroyed)
        for target in destroyed:
            del(undestroyed_targets[undestroyed_targets.index(target)])

        # Remove F-15s that were shot down
        destroyed_f_15s = destroyed_f_15s + f_15_down
        for f15 in f_15_down:
            del(f_15s_on_sortie[f_15s_on_sortie.index(f15[0])])

        # Have any F-15s out of fuel land to refuel
        f15_active = f_15s_on_sortie[:]
        for f15 in f15_active:
            if f15.empty():
                f15.refuel()
                f_15s = [f15] + f_15s
                del(f_15s_on_sortie[f_15s_on_sortie.index(f15)])

        # TODO Check if moving targets stopped and if they go down.

        # Build output that will be written as csv file
        output.append(stats(record, run, turn))

    print("Runtime: ",str(datetime.now()-start_time))
    lock.acquire()
    try:
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerows(output)
    finally:
        lock.release()

def stats(record: list, run: int, turn: int) -> list[str]:
    for line in record:
        out = [str(run),str(turn)]
        wpn_sys: WeaponSystem
        wpn: Weapon
        tgt: Target
        wpn_sys,tgt,wpn,dud,hit,fly,ada,engage = line

        out.append(str(wpn_sys))
        out.append(str(tgt))
        out.append(str(wpn))
        out.append(str(dud))
        out.append(str(hit))
        out.append(str(fly))
        out.append(str(ada))
        out.append(str(engage))
        out.append(str(wpn.cost))

        if hit and not dud:
            if wpn.area(tgt) > tgt.area:
                out.append(str(wpn.area(tgt) - tgt.area))
            else:
                out.append(str(None))
        elif not hit and not dud:
            out.append(str(wpn.area(tgt)))
        else:
            out.append(str(None))

    return(out)
