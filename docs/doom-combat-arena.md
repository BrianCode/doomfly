# DOOMFLY combat environment revision — 2026-09-05

The previous public run used ViZDoom's `defend_the_center`, a stock turn-and-shoot
scenario. The BCI also allowed movement. Recent rounds lasted the entire 60-game-
second cap, often ending with 100 health, no bullets and zero or a few kills.
Five melee enemies spawn initially and replacements depend on kills. Stationary
engine probes confirmed the stock monsters still worked; their absence from the
camera was not evidence that the spawner had stopped.

The initial replacement `combat-survival-v1` is an original 768×768 Doom room. Four
monsters start the round; every 105 engine tics another spawn is attempted,
up to eight alive. Spawn pads are fixed. Blocked pads are skipped. Monsters use
native Doom AI: alternating DoomImp (30 health, melee/projectile attacks) and
Zombieman (20 health, hitscan attacks). The engine targets the
player; it never supplies the player's commands. Corpses are removed after 20
game seconds to bound accumulation. The map uses the existing Freedoom assets.

Four ClipBox pickup pads give 50 bullets on physical contact, limited by the
normal 200-bullet capacity. Each pad returns eight game seconds after collection.
Killed monsters also drop a native Clip (dropped-ammo amount uses engine rules).
The round starts with 100 bullets and 100 health. There is no time limit or exit;
player death ends it. The broadcaster then resets only the game. The brain and
decoder are created once per broadcast run and continue across round resets.

The controller is unchanged: actual RGB frame → 3,335 inferred visual inputs →
the complete retained MaleCNS v1.0 graph → fixed DNp20/DNpe017 decoder → game
buttons. Observer-only USER1–4 fields report enemies alive, cumulative spawns,
ammo-box collections and boxes spawned. Enemy and item coordinates are never
inputs to the brain or decoder. Item counters exclude dropped Clip pickups.

This is still the no-plasticity baseline, with reward stimulation off. Persistent
neural state is not persistent learned memory. Death does not teach it anything.
The separate learning candidates have not passed scientific validation. Better
combat pacing, longer rounds or incidental ammo pickups establish no learning.
The current arena is a new environment, so performance cannot be pooled with
the original hour-long audit or blue-floor conditioning/survival results.

## Validation

`tests/test_doom_combat.py` runs real ViZDoom checks for initial state, damage and
death, reset counters/ammo, spawns, eight-enemy cap, survival beyond 60 seconds,
ammo contact/cooldown/recollection, actual shooting kills and collectible death
drops. Invulnerability, teleporting, inventory changes and monster removal are
explicit test fixtures used to isolate those mechanics. They are absent from
the production broadcaster. Asset hashes are checked against the shipped
manifest. Six environment tests and eight existing neural-boundary tests passed.

Source and generated WAD are in `doom/scenarios`; rebuild uses official
[ZDoom ACC](https://github.com/ZDoom/acc). The unchanged runtime is
[ViZDoom](https://vizdoom.farama.org/). The previous scenario is documented in
[ViZDoom's default environments](https://vizdoom.farama.org/environments/default/).

The deployment starts a new identified run. The previous run's final state and
audit segments are preserved locally before replacement. The research lab's
historical results remain unchanged and labeled as earlier experiments.


## V2 circling regression — 2026-09-05

V1 allowed surviving imps to occupy all eight enemy slots. Bullet-firing zombies
were killed, while slower projectile enemies survived and suppressed further
replacements. A separate fixed-controller probe reproduced five game minutes
of circling with eight DoomFlyImp objects. The public run also showed prolonged
circling at 58 health, seven kills and no ammunition. Per-type counts were not
available in that live v1 run, so its exact enemy composition was not measured.

`combat-survival-v2` caps each type at four. The three-second spawn attempt
alternates types when possible and fills the other type when one cap is reached.
Native attacks, infighting, movement, damage and collision remain in the engine.
No forced death, timer, anti-circling player policy, extra neural stimulation,
changed decoder gains, or learning was added. USER5 and USER6 expose current imp
and zombie counts to observers only; they are never read by the controller.
Vacancies may exist between attempts or while spawn pads are blocked.

`doom/arena_probe.py` tests a fixed command (turn 0.772, forward 19.226, attack on)
chosen from the observed loop before revising the arena. It drives isolated game
instances without a brain, using the same one-tic action boundary as production.
All eight seeds were paired across revisions; the first three informed diagnosis
and the final five were held out from that initial diagnosis. Four v1 runs were
still alive at the 300-game-second observation cap. All eight v2 runs died through
normal enemy attacks in 13.6–71.83 game seconds. The seed that first reproduced the
stalemate died at 42.8 seconds. This is a bounded environment regression, not proof
against every repetitive strategy or evidence that a fly learned. The live neural
trajectory can still circle; the fix prevents imps from monopolizing spawn slots.

Twenty environment, neural-boundary, broadcast and recovery tests passed. The
population test allows normal infighting vacancies and checks per-type bounds.
Exact WAD hashes, seed outcomes and limits are in `/arena-circling-review.json`.
The probe and paired evidence are included in the downloadable source archive.

The final v1 state, source assets, audit logs and final neural checkpoint were
preserved under `outputs/doom/circling-review` and `outputs/doom/spectator-v1`.
Switching revisions interrupts and censors the old unfinished round. V2 starts a
fresh identified baseline and fresh neural state, with its own audit/checkpoint
directory; incompatible checkpoints are not imported. Natural deaths within v2
continue to preserve neural state. These are separate environments and must not
be joined into a survival-improvement or learning curve.
