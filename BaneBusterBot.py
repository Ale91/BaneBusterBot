import os
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
METABOLICBOOST = UpgradeId.ZERGLINGMOVEMENTSPEED

os.environ["SC2PATH"] = '~/Data/StarCraftII/'
# os.environ["SC2PATH"] = '~/Games/starcraft-ii/drive_c/Program Files (x86)/StarCraft II/'


class BaneBusterBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.ITERATIONS_PER_MINUTE = 165
        self.do_something_after = 0
        self.use_model = use_model
        self.extractor_made = False
        self.metabolic_boost_started = False
        self.first_ov_done = False
        self.army_role = 'defense'
        self.drone_sent = False

        self.train_data = []

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")

    async def on_step(self,iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_units()
        await self.build_buildings()
        await self.inject()
        await self.attack()
        await self.intel()

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1],self.game_info.map_size[0], 3), np.uint8)
        self.game_info.map_size
        draw_dict = {
            HATCHERY: [7, (0, 255, 0)],
            OVERLORD: [2, (20, 235, 0)],
            DRONE: [1, (55, 200, 0)],
            EXTRACTOR: [2, (55, 200, 0)],
            SPAWNING_POOL: [3, (200, 100, 0)],
            BANELINGNEST: [3, (150, 150, 0)],
            ZERGLING: [2, (255, 100, 0)],
            QUEEN: [3, (255, 100, 0)],
            BANELING: [2, (255, 100, 0)],
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

        for obs in self.units(OVERLORD).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

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

    def need_overlords(self):
        if (len(self.units(OVERLORD)) == 1) and (self.supply_used == 13) and not self.first_ov_done:
            return True
        elif (len(self.units(OVERLORD)) == 2) and (len(self.units(DRONE)) == 19):
            return True
        elif (len(self.units(OVERLORD)) == 3) and (len(self.units(DRONE)) == 19) and (len(self.units(SPAWNING_POOL).ready) == 0):
            return True
        elif (self.supply_left < 3) and (len(self.units(OVERLORD).not_ready) == 0) and (len(self.units(DRONE)) > 18):
            return True
        else:
            return False

    async def build_units(self):
        for hatchery in self.units(HATCHERY).ready.noqueue:
            if (len(self.units(QUEEN)) < 1) and self.units(SPAWNING_POOL).ready.exists:
                if self.can_afford(QUEEN):
                    await self.do(hatchery.train(QUEEN))
        if len(self.units(LARVA)) > 0:
            if (len(self.units(DRONE)) < 18) and not self.need_overlords():
                if self.can_afford(DRONE):
                    if self.supply_left > 0:
                        await self.do(self.units(LARVA).random.train(DRONE))
                        print("DRONE")
            if self.need_overlords():
                if self.can_afford(OVERLORD):
                    await self.do(self.units(LARVA).random.train(OVERLORD))
                    print("OVERLORD")
                    self.first_ov_done = True
            else:
                if self.units(SPAWNING_POOL).ready.exists:
                    if self.can_afford(ZERGLING):
                        await self.do(self.units(LARVA).random.train(ZERGLING))

    async def inject(self):
        for queen in self.units(QUEEN).ready.idle:
            abilities = await self.get_available_abilities(queen)
            if queen.energy > 25:
                if AbilityId.EFFECT_INJECTLARVA in abilities:
                    await self.do(queen(AbilityId.EFFECT_INJECTLARVA, self.units(HATCHERY).ready.closest_to(queen)))

    async def build_buildings(self):
        # 2nd base
        if self.minerals > 175 and not self.drone_sent:
            bases = list(self.expansion_locations.keys())
            self.units(OVERLORD).first.move(bases[1])
            self.drone_sent = True
        if (len(self.units(HATCHERY)) < 2):
            if self.can_afford(HATCHERY):
                await self.expand_now() #TODO send drone before 300 minerals
        # Extractor
        if len(self.units(HATCHERY)) > 1:
            if not self.units(EXTRACTOR).exists and not self.extractor_made:
                if self.can_afford(EXTRACTOR):
                    vespene = self.state.vespene_geyser.closest_to(self.units(HATCHERY).ready.first)
                    if await self.can_place(EXTRACTOR,vespene.position):
                        drone = self.units(DRONE).closest_to(vespene.position)
                        await self.do(drone.build(EXTRACTOR,vespene))
                        self.extractor_made = True
        # Spawning Pool
        if not self.units(SPAWNING_POOL).exists:
            if (len(self.units(HATCHERY)) > 1) and self.units(EXTRACTOR).exists:
                if self.can_afford(SPAWNING_POOL):
                    for d in range(4, 15):
                        pos = self.units(HATCHERY).first.position.to2.towards(self.game_info.map_center, d)
                        if await self.can_place(SPAWNING_POOL, pos):
                            drone = self.workers.closest_to(pos)
                            await self.do(drone.build(SPAWNING_POOL, pos))
                            print("SPAWNINGPOOL ")
                            with open("log.txt", "a") as f:
                                if self.use_model:
                                    f.write("spawned spawning pool")
        #Metabolic Boost
        if self.units(SPAWNING_POOL).ready.exists and self.can_afford(METABOLICBOOST):
            if not self.metabolic_boost_started:
                await self.do(self.units(SPAWNING_POOL).ready.first.research(METABOLICBOOST))
                self.metabolic_boost_started = True


    def find_target(self, state):  # TODO update target while not idle to attack units first
        # if len(self.known_enemy_units) > 0:
        #     return random.choice(self.known_enemy_units)
        # elif len(self.known_enemy_structures) > 0:
        #     return random.choice(self.known_enemy_structures)
        # else:
        #     return self.enemy_start_locations[0]
        return self.enemy_start_locations[0]

    async def attack(self): # TODO BANELINGS
        if self.units(ZERGLING).idle.exists:
            close_idles = self.units(ZERGLING).idle.closer_than(8, self.units(ZERGLING).idle.center)
            if close_idles.amount > 15:
                for x in close_idles:
                    await self.do(x.attack(self.find_target(self.state)))
            far_idles = self.units(ZERGLING).idle.further_than(8, self.units(ZERGLING).idle.center)
            for z in far_idles:
                if (self.units(ZERGLING) - self.units(ZERGLING).idle).exists:
                    await self.do(z.move((self.units(ZERGLING) - self.units(ZERGLING).idle).center))
                else:
                    await self.do(z.move(self.units(ZERGLING).center))


        # TODO ATTACK MODEL

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        # if game_result == sc2.Result.Victory:
        #     np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))


run_game(maps.get("CatalystLE"), [
    # Human(Race.Terran),
    Bot(Race.Zerg, BaneBusterBot(use_model=False)),
    Computer(Race.Terran, Difficulty.CheatInsane),
    ], realtime=False, save_replay_as="lastReplay.SC2Replay")
