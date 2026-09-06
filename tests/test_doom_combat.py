"""Real-engine environment QA. Programmed actions/cheats here are test fixtures,
never fly behavior, production controls or evidence of learning.
"""
import json
import hashlib
from pathlib import Path

import pytest
import vizdoom as vzd
from doom.game import Game
from doom.combat_arena import SCENARIOS

IDLE = {'turn': 0, 'forward': 0, 'attack': False}


@pytest.fixture
def arena():
    game = Game(scenario='combat_survival')
    yield game
    game.close()


def step(game, tics, action=IDLE):
    for _ in range(tics):
        game.act(action)


def test_spawn_attack_death_and_fresh_arena(arena):
    initial = arena.observation()
    assert (initial['health'], initial['ammo'], initial['enemies']) == (100, 100, 4)
    while not arena.observation()['finished'] and arena.tick < 35 * 30:
        arena.act(IDLE)
    final = arena.observation()
    assert final['finished'] and final['health'] <= 0
    assert final['enemies_spawned'] >= initial['enemies_spawned']
    arena.new_episode()
    reset = arena.observation()
    assert reset['episode'] == 2 and reset['tick'] == 0 and not reset['finished']
    assert (reset['health'], reset['ammo'], reset['enemies']) == (100, 100, 4)
    assert reset['ammo_pickups'] == 0


def test_no_time_limit_and_bounded_enemy_population(arena):
    # Invulnerability isolates the old timeout failure from actual death.
    arena.game.send_game_command('god')
    step(arena, 35 * 65)
    state = arena.observation()
    assert arena.game.get_episode_timeout() == 0
    assert not state['finished'] and state['health'] == 100
    # Native monster infighting can leave vacancies between spawn attempts.
    assert 0 < state['enemies'] <= 8 and state['enemies_spawned'] >= 8
    assert 0 <= state['imps'] <= 4 and 0 <= state['zombies'] <= 4
    assert state['imps'] + state['zombies'] == state['enemies']


def test_ammo_requires_contact_and_respawns_after_collection(arena):
    arena.game.send_game_command('god')
    step(arena, 100)
    assert arena.observation()['ammo'] == 100
    arena.game.send_game_command('warp 160 160')
    step(arena, 5, {**IDLE, 'forward': 1})
    state = arena.observation()
    assert state['ammo'] == 150 and state['ammo_pickups'] == 1
    arena.game.send_game_command('warp 0 0')
    step(arena, 250)
    assert arena.observation()['ammo_spawned'] == 4
    step(arena, 40)
    assert arena.observation()['ammo_spawned'] == 5
    arena.game.send_game_command('warp 160 160')
    step(arena, 5, {**IDLE, 'forward': 1})
    assert arena.observation()['ammo'] == 200
    assert arena.observation()['ammo_pickups'] == 2


def test_shots_kill_and_replacement_monsters_arrive(arena):
    arena.game.send_game_command('god')
    step(arena, 350, {**IDLE, 'attack': True})
    state = arena.observation()
    assert state['kills'] > 0 and state['ammo'] < 100
    assert state['enemies_spawned'] > 4
    assert state['enemies'] == state['enemies_spawned'] - state['kills']


def test_enemy_death_drops_collectible_ammo():
    # Object inspection is deliberately enabled only in this engine test.
    game = vzd.DoomGame()
    game.load_config(str(SCENARIOS/'combat_survival.cfg'))
    game.set_doom_scenario_path(str((SCENARIOS/'combat_survival.wad').resolve()))
    game.set_doom_game_path(str(Path(vzd.__file__).parent/'freedoom2.wad'))
    game.set_objects_info_enabled(True)
    game.set_window_visible(False)
    game.set_available_buttons([vzd.Button.MOVE_FORWARD])
    game.init()
    try:
        game.new_episode()
        game.send_game_command('god')
        game.send_game_command('take Clip 999')
        game.send_game_command('kill monsters')
        game.make_action([False], 60)
        clips = [o for o in game.get_state().objects if o.name == 'Clip']
        assert len(clips) == 4
        assert game.get_game_variable(vzd.GameVariable.USER1) == 0
        clip = clips[0]
        game.send_game_command(f'warp {clip.position_x} {clip.position_y}')
        game.make_action([True], 2)
        assert game.get_game_variable(vzd.GameVariable.AMMO2) > 0
    finally:
        game.close()


def test_shipped_combat_assets_match_manifest():
    manifest = json.loads((SCENARIOS/'combat_survival.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = SCENARIOS/name if name != 'combat_arena.py' else SCENARIOS.parent/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_both_enemy_types_remain_bounded_during_combat(arena):
    arena.game.send_game_command('god')
    for _ in range(35 * 90):
        arena.act({'turn': .772, 'forward': 19.226, 'attack': True})
        state = arena.observation()
        assert 0 <= state['imps'] <= 4 and 0 <= state['zombies'] <= 4
        assert state['imps'] + state['zombies'] == state['enemies']
    assert state['enemies_spawned'] > 8


def test_reproduced_v1_circling_stalemate_is_not_safe_in_v2():
    from doom.arena_probe import probe
    result = probe(SCENARIOS/'combat_survival.wad', 41029)
    assert result['dead'], 'The v1 fixed-circle stalemate must not survive the observation cap'
