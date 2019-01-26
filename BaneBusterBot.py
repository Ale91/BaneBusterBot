import time
import keras
import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer, Human
from sc2.constants import UnitTypeId, AbilityId, BuffId, EffectId, UpgradeId
import random
import numpy as np
import cv2

HEADLESS = False

# PROBE = UnitTypeId.PROBE
# NEXUS = UnitTypeId.NEXUS
# PYLON = UnitTypeId.PYLON
# ASSIMILATOR = UnitTypeId.ASSIMILATOR
# GATEWAY = UnitTypeId.GATEWAY
# CYBERNETICSCORE = UnitTypeId.CYBERNETICSCORE
# STALKER = UnitTypeId.STALKER
# VOIDRAY = UnitTypeId.VOIDRAY
# STARGATE = UnitTypeId.STARGATE
# ZEALOT = UnitTypeId.ZEALOT
# ROBOTICSFACILITY = UnitTypeId.ROBOTICSFACILITY
# OBSERVER = UnitTypeId.OBSERVER
# IMMORTAL = UnitTypeId.IMMORTAL
DRONE = UnitTypeId.DRONE
LARVA = UnitTypeId.LARVA
QUEEN = UnitTypeId.QUEEN
EXTRACTOR = UnitTypeId.EXTRACTOR
HATCHERY = UnitTypeId.HATCHERY
ZERGLING = UnitTypeId.ZERGLING
BANELING = UnitTypeId.BANELING
SPAWNING_POOL = UnitTypeId.SPAWNINGPOOL
BANELINGNEST = UnitTypeId.BANELINGNEST
OVERLORD = UnitTypeId.OVERLORD

# os.environ["SC2PATH"] = '/starcraftstuff/StarCraftII/'


class BaneBusterBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.ITERATIONS_PER_MINUTE = 165
        self.do_something_after = 0
        self.use_model = use_model

        self.train_data = []

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")

    async def on_step(self,iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_overlords()
        await self.build_extractor()
        await self.inject()
        await self.expand()
        await self.offensive_buildings()
        await self.offensive_force()
        await self.attack()
        await self.intel()
        await self.scout()

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1],self.game_info.map_size[0], 3), np.uint8)
        self.game_info.map_size
        draw_dict = {
            # NEXUS: [7, (0, 255, 0)],
            # PYLON: [2, (20, 235, 0)],
            # PROBE: [1, (55, 200, 0)],
            # ASSIMILATOR: [2, (55, 200, 0)],
            # GATEWAY: [3, (200, 100, 0)],
            # CYBERNETICSCORE: [3, (150, 150, 0)],
            # STARGATE: [3, (255, 0, 0)],
            # VOIDRAY: [3, (255, 100, 0)],
            # ROBOTICSFACILITY: [3, (215, 155, 0)],
            HATCHERY: [7, (0, 255, 0)],
            OVERLORD: [2, (20, 235, 0)],
            DRONE: [1, (55, 200, 0)],
            EXTRACTOR: [2, (55, 200, 0)],
            SPAWNING_POOL: [3, (200, 100, 0)],
            BANELINGNEST: [3, (150, 150, 0)],
            ZERGLING: [3, (255, 100, 0)],
            QUEEN: [3, (255, 100, 0)],
            BANELING: [3, (255, 100, 0)],
        }
        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]),int(pos[1])), draw_dict[unit_type][0],draw_dict[unit_type][1], -1)

        # for obs in self.units(OBSERVER).ready:
        #     pos = obs.position
        #     cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / (self.supply_cap+1)
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(ZERGLING))+len(self.units(BANELING)) / (self.supply_cap-self.supply_left)+1
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
        cv2.imshow('Intel', resized)
        cv2.waitKey(1)

    def check_overlords(self):
        if (len(self.units(OVERLORD)) == 0) and (len(self.units(DRONE)) > 12):
            return False
        elif (len(self.units(OVERLORD)) == 1) and (len(self.units(DRONE)) > 18):
            return False
        else:
            return True
        #TODO

    async def build_units(self):
        #for hatchery in self.units(HATCHERY).ready.noqueue:
        if len(self.units(LARVA)) > 0:
            if (len(self.units(DRONE)) < 19) and self.check_overlords():
                if self.can_afford(self, DRONE):
                    if self.supply_left > 0:
                        await self.do(LARVA.random.train(DRONE))
            if len(self.units(DRONE)) > 8:

        #if self.can_afford(PROBE) and len(self.units(PROBE)) < 60:
            #if len(self.units(PROBE)) < self.units(NEXUS).amount*22:
                #await self.do(nexus.train(PROBE))

    # async def build_overlords(self):
    #     if self.supply_left < 5 and not self.already_pending(PYLON):
    #         nexuses = self.units(NEXUS).ready
    #         if nexuses.exists:
    #             if self.can_afford(PYLON):
    #                 if self.FIRST_PYLON:
    #                     await self.build(PYLON, near=nexuses.first, placement_step=7)
    #                     self.FIRST_PYLON = False
    #                 else:
    #                     await self.build(PYLON, near=self.units(PYLON).random, placement_step=5)
    #
    #     if len(self.units(GATEWAY).ready) > 3 and self.supply_left < 13 and len(self.units(PYLON).not_ready) <2:
    #                 await self.build(PYLON, near=self.units(PYLON).random, placement_step=5)
    #     if len(self.units(GATEWAY).ready) > 6 and self.supply_left < 20 and len(self.units(PYLON).not_ready) <3:
    #                 await self.build(PYLON, near=self.units(PYLON).random, placement_step=5)

    async def inject(self):
        for queen in self.units(QUEEN).ready:
            abilities = await self.get_available_abilities(queen)
            if queen.energy > 50:
                if AbilityId.EFFECT_INJECTLARVA in abilities:
                    if not HATCHERY.has_buff(BuffId.):
                        await self.do(queen(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

    async def build_extractor(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(10.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if ((len(self.units(NEXUS)) < 3)  and ((self.iteration / self.ITERATIONS_PER_MINUTE) < 6)) or ((len(self.units(NEXUS)) < 4)  and ((self.iteration / self.ITERATIONS_PER_MINUTE) < 8)) or ((len(self.units(NEXUS)) < 5) and ((self.iteration / self.ITERATIONS_PER_MINUTE) > 8)):
            if self.can_afford(NEXUS):
                await self.expand_now()

    async def offensive_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists:
                if not self.units(CYBERNETICSCORE):
                    if self.can_afford(CYBERNETICSCORE):
                        await self.build(CYBERNETICSCORE, near=pylon)   # Closer_than to nexus? and farther_than from minerals? TODO
            if  ((len(self.units(GATEWAY)) < 1)  and (len(self.units(NEXUS)) > 0)) or ((len(self.units(GATEWAY)) < 3) and (len(self.units(NEXUS)) > 1)) or ((len(self.units(GATEWAY)) < 7)  and (len(self.units(NEXUS)) > 2)) or ((len(self.units(GATEWAY)) < 12)  and (len(self.units(NEXUS)) > 3)):
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

    # async def offensive_force(self):
        # for larvae in self.units(L).ready.noqueue :
        #     if (self.can_afford(STALKER) and self.supply_left > 1) and (not self.units(STALKER).amount > self.units(ZEALOT).amount):
        #         await self.do(gw.train(STALKER))
        #     if (self.can_afford(ZEALOT) and self.supply_left > 1) and (not self.units(ZEALOT).amount > self.units(STALKER).amount+2):
        #         await self.do(gw.train(ZEALOT))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        if len(self.units(ZEALOT).idle)+len(self.units(STALKER).idle) > 0:
            target = False
            if self.iteration > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])
                    # print('prediction: ',choice)

                    choice_dict = {0: "No Attack!",
                                   1: "Attack close to our nexus!",
                                   2: "Attack Enemy Structure!",
                                   3: "Attack Eneemy Start!"}

                    print("Choice #{}:{}".format(choice, choice_dict[choice]))
                else:
                    choice = random.randrange(0, 4)
                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    # attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(ZEALOT).idle+self.units(STALKER).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y, self.flipped])
        # if self.supply_used > 180 and self.units(STALKER).idle.amount+self.units(ZEALOT).idle.amount > 15:
        #         for s in self.units(STALKER).idle+self.units(ZEALOT).idle:
        #             await self.do(s.attack(self.find_target(self.state)))
        # if self.units(STALKER).amount+self.units(ZEALOT).amount > 5:
        #     if len(self.known_enemy_units) > 0:
        #         for s in self.units(STALKER).idle+self.units(ZEALOT).idle:
        #             await self.do(s.attack(random.choice(self.known_enemy_units)))

    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))

        else:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        if game_result == sc2.Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))


run_game(maps.get("AbyssalReefLE"), [
    # Human(Race.Terran),
    Bot(Race.Protoss, Zeabot(use_model=False)),
    Computer(Race.Terran, Difficulty.Hard),
    ], realtime=True)
