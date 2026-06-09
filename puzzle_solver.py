#!/usr/bin/env python3
"""
Solver for the 4x4 sliding puzzle described in the prompt.

There are 16 numbered tiles in a 4x4 grid and one blank slot, 00, outside the
grid. The outside slot is connected to the bottom-right grid cell:

    11 12 13 14
    21 22 23 24
    31 32 33 34
    41 42 43 44
                00

A move swaps 00 with one neighbouring tile. This script solves any valid,
reachable scramble for that board.
"""

from __future__ import annotations

import argparse
import heapq
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from math import inf
from typing import Iterable


ROWS = 4
COLS = 4
POCKET = ROWS * COLS
BLANK = 0

TILE_LABELS = ["00"] + [f"{row}{col}" for row in range(1, ROWS + 1) for col in range(1, COLS + 1)]
LABEL_TO_TILE = {label: tile for tile, label in enumerate(TILE_LABELS)}
GOAL = tuple(range(1, ROWS * COLS + 1)) + (BLANK,)
GOAL_POS = {tile: tile - 1 for tile in range(1, ROWS * COLS + 1)}
GOAL_POS[BLANK] = POCKET
GOAL_ROW_COL = {tile: divmod(pos, COLS) for tile, pos in GOAL_POS.items() if pos < POCKET}


def build_adjacency() -> list[tuple[int, ...]]:
    adjacency: list[list[int]] = [[] for _ in range(POCKET + 1)]

    for row in range(ROWS):
        for col in range(COLS):
            pos = row * COLS + col
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    adjacency[pos].append(nr * COLS + nc)

    adjacency[POCKET].append(POCKET - 1)
    adjacency[POCKET - 1].append(POCKET)
    return [tuple(neighbours) for neighbours in adjacency]


ADJACENCY = build_adjacency()


def build_distances() -> list[list[int]]:
    distances: list[list[int]] = []
    for start in range(POCKET + 1):
        dist = [10**9] * (POCKET + 1)
        dist[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            here = queue.popleft()
            for nxt in ADJACENCY[here]:
                if dist[nxt] == 10**9:
                    dist[nxt] = dist[here] + 1
                    queue.append(nxt)
        distances.append(dist)
    return distances


DISTANCES = build_distances()


def build_colours() -> list[int]:
    colours = [0] * (POCKET + 1)
    for pos in range(POCKET):
        row, col = divmod(pos, COLS)
        colours[pos] = (row + col) % 2
    colours[POCKET] = 1 - colours[POCKET - 1]
    return colours


COLOURS = build_colours()


@dataclass
class SolveStats:
    nodes: int = 0
    iterations: int = 0
    seconds: float = 0.0
    algorithm: str = "IDA*"


class PuzzleError(ValueError):
    pass


def normalise_token(token: str) -> str:
    token = token.strip()
    if token == "0":
        return "00"
    return token


def parse_state(tokens: Iterable[str]) -> tuple[int, ...]:
    cleaned = [normalise_token(token) for token in tokens if token.strip()]

    if len(cleaned) != POCKET + 1:
        raise PuzzleError(f"Expected 17 values: 16 grid tiles plus the outside slot. Got {len(cleaned)}.")

    unknown = [token for token in cleaned if token not in LABEL_TO_TILE]
    if unknown:
        raise PuzzleError(f"Unknown tile label(s): {', '.join(unknown)}")

    state = tuple(LABEL_TO_TILE[token] for token in cleaned)
    expected = set(range(POCKET + 1))
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        duplicates = sorted(tile for tile in expected if state.count(tile) > 1)
        missing_labels = ", ".join(TILE_LABELS[tile] for tile in missing) or "none"
        duplicate_labels = ", ".join(TILE_LABELS[tile] for tile in duplicates) or "none"
        raise PuzzleError(f"Invalid puzzle. Missing: {missing_labels}. Duplicate: {duplicate_labels}.")

    return state


def parse_state_text(text: str) -> tuple[int, ...]:
    return parse_state(text.replace(",", " ").split())


def format_position(pos: int) -> str:
    if pos == POCKET:
        return "outside slot"
    row, col = divmod(pos, COLS)
    return f"row {row + 1}, col {col + 1}"


def format_state(state: tuple[int, ...]) -> str:
    lines = []
    for row in range(ROWS):
        start = row * COLS
        lines.append(" ".join(f"{TILE_LABELS[tile]:>2}" for tile in state[start : start + COLS]))
    lines.append(f"{'':>11}{TILE_LABELS[state[POCKET]]:>2}  outside")
    return "\n".join(lines)


def distance_heuristic(state: tuple[int, ...]) -> int:
    total = 0
    for pos, tile in enumerate(state):
        if tile != BLANK:
            total += DISTANCES[pos][GOAL_POS[tile]]
    return total


def count_inversions(values: list[int]) -> int:
    total = 0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if left > right:
                total += 1
    return total


def linear_conflict_heuristic(state: tuple[int, ...]) -> int:
    conflicts = 0

    for row in range(ROWS):
        goal_cols = []
        for col in range(COLS):
            tile = state[row * COLS + col]
            if tile == BLANK:
                continue
            goal_row, goal_col = GOAL_ROW_COL[tile]
            if goal_row == row:
                goal_cols.append(goal_col)
        conflicts += count_inversions(goal_cols)

    for col in range(COLS):
        goal_rows = []
        for row in range(ROWS):
            tile = state[row * COLS + col]
            if tile == BLANK:
                continue
            goal_row, goal_col = GOAL_ROW_COL[tile]
            if goal_col == col:
                goal_rows.append(goal_row)
        conflicts += count_inversions(goal_rows)

    return conflicts * 2


def heuristic(state: tuple[int, ...]) -> int:
    return distance_heuristic(state) + linear_conflict_heuristic(state)


def permutation_parity(state: tuple[int, ...]) -> int:
    goal_order = [GOAL_POS[tile] for tile in state]
    inversions = 0
    for i, left in enumerate(goal_order):
        for right in goal_order[i + 1 :]:
            if left > right:
                inversions ^= 1
    return inversions


def is_solvable(state: tuple[int, ...]) -> bool:
    blank_pos = state.index(BLANK)
    blank_colour_change = COLOURS[blank_pos] ^ COLOURS[POCKET]
    return permutation_parity(state) == blank_colour_change


def reconstruct_path(
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], int]],
    end_state: tuple[int, ...],
) -> list[int]:
    moves: list[int] = []
    here = end_state
    while here in parent:
        previous, tile = parent[here]
        moves.append(tile)
        here = previous
    moves.reverse()
    return moves


def ordered_blank_moves(board: list[int], blank_pos: int, current_h: int) -> list[tuple[int, int, int, int]]:
    moves = []
    for next_blank_pos in ADJACENCY[blank_pos]:
        tile = board[next_blank_pos]
        old_distance = DISTANCES[next_blank_pos][GOAL_POS[tile]]
        new_distance = DISTANCES[blank_pos][GOAL_POS[tile]]
        delta_h = new_distance - old_distance
        board[blank_pos], board[next_blank_pos] = tile, BLANK
        next_h = heuristic(tuple(board))
        board[blank_pos], board[next_blank_pos] = BLANK, tile
        moves.append((delta_h, next_blank_pos, tile, next_h))
    moves.sort(key=lambda move: (move[3], move[0]))
    return moves


def solve_ida_star(state: tuple[int, ...], max_seconds: float | None = None) -> tuple[list[int], SolveStats]:
    if not is_solvable(state):
        raise PuzzleError("This scramble is valid but impossible to solve from the target position.")

    if state == GOAL:
        return [], SolveStats(seconds=0.0, algorithm="IDA*")

    start_time = time.monotonic()
    stats = SolveStats(algorithm="IDA*")
    board = list(state)
    path: list[int] = []
    seen = {state}
    bound = heuristic(state)

    def search(blank_pos: int, previous_blank_pos: int, depth: int, current_h: int, limit: int) -> int | None:
        stats.nodes += 1

        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            raise TimeoutError(f"Stopped after {max_seconds:g} seconds. Try a higher --max-seconds value.")

        score = depth + current_h
        if score > limit:
            return score

        if current_h == 0:
            return None

        next_limit = inf
        moves = ordered_blank_moves(board, blank_pos, current_h)
        for delta_h, next_blank_pos, tile, _next_h in moves:
            if next_blank_pos == previous_blank_pos:
                continue
            board[blank_pos], board[next_blank_pos] = tile, BLANK
            key = tuple(board)

            if key not in seen:
                seen.add(key)
                path.append(tile)
                result = search(next_blank_pos, blank_pos, depth + 1, current_h + delta_h, limit)
                if result is None:
                    return None
                next_limit = min(next_limit, result)
                path.pop()
                seen.remove(key)

            board[blank_pos], board[next_blank_pos] = BLANK, tile

        return next_limit

    blank = state.index(BLANK)
    while True:
        stats.iterations += 1
        result = search(blank, -1, 0, heuristic(tuple(board)), bound)
        if result is None:
            stats.seconds = time.monotonic() - start_time
            return path[:], stats
        if result == inf:
            raise PuzzleError("No solution found.")
        bound = int(result)


def solve_a_star_closed(
    state: tuple[int, ...],
    max_seconds: float | None = None,
    max_nodes: int | None = None,
) -> tuple[list[int], SolveStats]:
    if not is_solvable(state):
        raise PuzzleError("This scramble is valid but impossible to solve from the target position.")

    if state == GOAL:
        return [], SolveStats(seconds=0.0, iterations=1, algorithm="A* (closed set)")

    start_time = time.monotonic()
    stats = SolveStats(iterations=1, algorithm="A* (closed set)")
    start_h = heuristic(state)
    heap: list[tuple[int, int, int, int, tuple[int, ...], int]] = []
    tie = count()
    heapq.heappush(heap, (start_h, start_h, 0, next(tie), state, state.index(BLANK)))

    best_g: dict[tuple[int, ...], int] = {state: 0}
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    closed: set[tuple[int, ...]] = set()

    while heap:
        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            raise TimeoutError(f"Stopped after {max_seconds:g} seconds. Try a higher --max-seconds value.")

        _f, current_h, current_g, _tie, current, blank = heapq.heappop(heap)
        if current in closed:
            continue
        if current_g != best_g.get(current):
            continue

        closed.add(current)
        stats.nodes += 1

        if max_nodes is not None and stats.nodes > max_nodes:
            raise TimeoutError(f"A* stopped after searching {max_nodes:,} states.")

        if current_h == 0:
            stats.seconds = time.monotonic() - start_time
            return reconstruct_path(parent, current), stats

        board = list(current)
        for _delta_h, next_blank, tile, next_h in ordered_blank_moves(board, blank, current_h):
            board[blank], board[next_blank] = tile, BLANK
            next_state = tuple(board)
            board[blank], board[next_blank] = BLANK, tile

            if next_state in closed:
                continue

            next_g = current_g + 1
            if next_g < best_g.get(next_state, 10**9):
                best_g[next_state] = next_g
                parent[next_state] = (current, tile)
                heapq.heappush(heap, (next_g + next_h, next_h, next_g, next(tie), next_state, next_blank))

    raise PuzzleError("No solution found.")


def solve_weighted_a_star(
    state: tuple[int, ...],
    max_seconds: float | None = None,
    max_nodes: int | None = 1_500_000,
    weight: float = 3.0,
) -> tuple[list[int], SolveStats]:
    if not is_solvable(state):
        raise PuzzleError("This scramble is valid but impossible to solve from the target position.")

    if state == GOAL:
        return [], SolveStats(seconds=0.0, iterations=1, algorithm="Fast weighted A*")

    start_time = time.monotonic()
    stats = SolveStats(iterations=1, algorithm="Fast weighted A*")
    start_h = heuristic(state)
    heap: list[tuple[float, int, int, int, tuple[int, ...], int]] = []
    tie = count()
    heapq.heappush(heap, (start_h * weight, start_h, 0, next(tie), state, state.index(BLANK)))

    best_g: dict[tuple[int, ...], int] = {state: 0}
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    closed: set[tuple[int, ...]] = set()

    while heap:
        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            raise TimeoutError(f"Fast search stopped after {max_seconds:g} seconds.")

        _f, current_h, current_g, _tie, current, blank = heapq.heappop(heap)
        if current in closed:
            continue
        if current_g != best_g.get(current):
            continue

        closed.add(current)
        stats.nodes += 1

        if max_nodes is not None and stats.nodes > max_nodes:
            raise TimeoutError(f"Fast search stopped after searching {max_nodes:,} states.")

        if current_h == 0:
            stats.seconds = time.monotonic() - start_time
            return reconstruct_path(parent, current), stats

        board = list(current)
        for _delta_h, next_blank, tile, next_h in ordered_blank_moves(board, blank, current_h):
            board[blank], board[next_blank] = tile, BLANK
            next_state = tuple(board)
            board[blank], board[next_blank] = BLANK, tile

            if next_state in closed:
                continue

            next_g = current_g + 1
            if next_g < best_g.get(next_state, 10**9):
                best_g[next_state] = next_g
                parent[next_state] = (current, tile)
                heapq.heappush(heap, (next_g + (weight * next_h), next_h, next_g, next(tie), next_state, next_blank))

    raise PuzzleError("No solution found.")


def solve_bfs(
    state: tuple[int, ...],
    max_seconds: float | None = None,
    max_nodes: int | None = 250_000,
) -> tuple[list[int], SolveStats]:
    if not is_solvable(state):
        raise PuzzleError("This scramble is valid but impossible to solve from the target position.")

    if state == GOAL:
        return [], SolveStats(seconds=0.0, iterations=1, algorithm="BFS")

    start_time = time.monotonic()
    stats = SolveStats(iterations=1, algorithm="BFS")
    queue: deque[tuple[int, ...]] = deque([state])
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
    seen = {state}

    while queue:
        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            raise TimeoutError(f"Stopped after {max_seconds:g} seconds. Try a higher --max-seconds value.")

        current = queue.popleft()
        stats.nodes += 1
        if max_nodes is not None and stats.nodes > max_nodes:
            raise TimeoutError(f"BFS stopped after searching {max_nodes:,} states.")

        blank = current.index(BLANK)
        board = list(current)
        for next_blank in ADJACENCY[blank]:
            tile = board[next_blank]
            board[blank], board[next_blank] = tile, BLANK
            next_state = tuple(board)
            board[blank], board[next_blank] = BLANK, tile

            if next_state in seen:
                continue
            parent[next_state] = (current, tile)
            if next_state == GOAL:
                stats.seconds = time.monotonic() - start_time
                return reconstruct_path(parent, next_state), stats
            seen.add(next_state)
            queue.append(next_state)

    raise PuzzleError("No solution found.")


def normalise_algorithm(algorithm: str | None) -> str:
    value = (
        (algorithm or "auto")
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
    )
    aliases = {
        "ida": "ida-star",
        "ida*": "ida-star",
        "ida-star": "ida-star",
        "a*": "a-star-closed",
        "astar": "a-star-closed",
        "a-star": "a-star-closed",
        "a*-closed": "a-star-closed",
        "astar-closed": "a-star-closed",
        "a-star-closed": "a-star-closed",
        "a*-closed-set": "a-star-closed",
        "a-star-closed-set": "a-star-closed",
        "fast": "fast",
        "weighted": "fast",
        "weighted-a-star": "fast",
        "weighted-astar": "fast",
        "fast-weighted-a-star": "fast",
        "bfs": "bfs",
        "strategic": "auto",
        "strategically": "auto",
        "auto": "auto",
    }
    if value not in aliases:
        raise PuzzleError(f"Unknown algorithm: {algorithm}")
    return aliases[value]


def solve(
    state: tuple[int, ...],
    max_seconds: float | None = None,
    algorithm: str | None = "ida-star",
    max_nodes: int | None = None,
) -> tuple[list[int], SolveStats]:
    algorithm_id = normalise_algorithm(algorithm)

    if algorithm_id == "ida-star":
        return solve_ida_star(state, max_seconds=max_seconds)
    if algorithm_id == "a-star-closed":
        return solve_a_star_closed(state, max_seconds=max_seconds, max_nodes=max_nodes)
    if algorithm_id == "fast":
        return solve_weighted_a_star(state, max_seconds=max_seconds, max_nodes=max_nodes or 1_500_000)
    if algorithm_id == "bfs":
        return solve_bfs(state, max_seconds=max_seconds, max_nodes=max_nodes or 250_000)

    try:
        cap_seconds = min(max_seconds, 6.0) if max_seconds is not None else 6.0
        return solve_weighted_a_star(state, max_seconds=cap_seconds, max_nodes=max_nodes or 900_000)
    except TimeoutError:
        pass

    try:
        cap_seconds = min(max_seconds, 5.0) if max_seconds is not None else 5.0
        return solve_a_star_closed(state, max_seconds=cap_seconds, max_nodes=max_nodes or 500_000)
    except TimeoutError:
        return solve_ida_star(state, max_seconds=max_seconds)


def apply_moves(state: tuple[int, ...], moves: Iterable[int]) -> list[tuple[int, ...]]:
    board = list(state)
    states = [tuple(board)]
    for tile in moves:
        blank = board.index(BLANK)
        tile_pos = board.index(tile)
        if tile_pos not in ADJACENCY[blank]:
            raise PuzzleError(f"Move {TILE_LABELS[tile]} is not legal from this state.")
        board[blank], board[tile_pos] = board[tile_pos], board[blank]
        states.append(tuple(board))
    return states


def move_direction(tile_pos: int, blank_pos: int) -> str:
    if tile_pos == POCKET - 1 and blank_pos == POCKET:
        return "down"
    if tile_pos == POCKET and blank_pos == POCKET - 1:
        return "up"
    if tile_pos < POCKET and blank_pos < POCKET:
        tile_row, tile_col = divmod(tile_pos, COLS)
        blank_row, blank_col = divmod(blank_pos, COLS)
        row_delta = blank_row - tile_row
        col_delta = blank_col - tile_col
        if row_delta == -1 and col_delta == 0:
            return "up"
        if row_delta == 1 and col_delta == 0:
            return "down"
        if row_delta == 0 and col_delta == -1:
            return "left"
        if row_delta == 0 and col_delta == 1:
            return "right"
    raise PuzzleError(f"No direction for move from {format_position(tile_pos)} to {format_position(blank_pos)}.")


def describe_moves(state: tuple[int, ...], moves: Iterable[int]) -> list[dict[str, str | int]]:
    board = list(state)
    steps: list[dict[str, str | int]] = []
    for step_number, tile in enumerate(moves, start=1):
        blank = board.index(BLANK)
        tile_pos = board.index(tile)
        if tile_pos not in ADJACENCY[blank]:
            raise PuzzleError(f"Move {TILE_LABELS[tile]} is not legal from this state.")
        direction = move_direction(tile_pos, blank)
        steps.append(
            {
                "step": step_number,
                "tile": TILE_LABELS[tile],
                "direction": direction,
                "from": format_position(tile_pos),
                "to": format_position(blank),
                "text": f"{TILE_LABELS[tile]} {direction}",
            }
        )
        board[blank], board[tile_pos] = board[tile_pos], board[blank]
    return steps


def random_scramble(steps: int, seed: int | None = None) -> tuple[int, ...]:
    rng = random.Random(seed)
    board = list(GOAL)
    blank = POCKET
    previous_blank = -1

    for _ in range(steps):
        choices = [pos for pos in ADJACENCY[blank] if pos != previous_blank]
        if not choices:
            choices = list(ADJACENCY[blank])
        next_blank = rng.choice(choices)
        board[blank], board[next_blank] = board[next_blank], board[blank]
        previous_blank, blank = blank, next_blank

    return tuple(board)


def print_solution(start: tuple[int, ...], moves: list[int], stats: SolveStats, show_states: bool) -> None:
    print("Start:")
    print(format_state(start))
    print()

    if not moves:
        print("Already solved.")
        return

    print(f"Solution length: {len(moves)} move(s)")
    print(f"Moves: {' '.join(TILE_LABELS[tile] for tile in moves)}")
    print(f"Directions: {', '.join(step['text'] for step in describe_moves(start, moves))}")
    iteration_label = "iteration" if stats.iterations == 1 else "iterations"
    print(
        f"Algorithm: {stats.algorithm}. "
        f"Searched {stats.nodes:,} node(s) in {stats.iterations} {iteration_label}, {stats.seconds:.3f}s"
    )

    if show_states:
        print()
        for step, state in enumerate(apply_moves(start, moves)):
            if step:
                print(f"After move {step}:")
            print(format_state(state))
            print()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the 4x4 tile puzzle with 00 as the outside slot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python puzzle_solver.py --state \"11 12 13 14 21 22 23 24 31 32 33 34 41 42 00 43 44\"\n"
            "  python puzzle_solver.py --scramble 40 --seed 7 --show-states\n"
            "  python puzzle_solver.py 11 12 13 14 21 22 23 24 31 32 33 34 41 42 00 43 44\n"
        ),
    )
    parser.add_argument("tokens", nargs="*", help="17 tile labels in row-major order, with the outside slot last")
    parser.add_argument("--state", help="State as 17 labels: 16 grid values followed by the outside slot value")
    parser.add_argument("--scramble", type=int, help="Generate this many legal random moves from the solved state")
    parser.add_argument("--seed", type=int, help="Seed for --scramble")
    parser.add_argument(
        "--algorithm",
        choices=("auto", "fast", "ida-star", "a-star-closed", "bfs"),
        default="fast",
        help="Search algorithm to use. auto tries fast weighted A*, exact A*, then IDA*.",
    )
    parser.add_argument("--show-states", action="store_true", help="Print the board after every solution move")
    parser.add_argument("--max-seconds", type=float, help="Stop if solving takes longer than this many seconds")
    return parser


def main(argv: list[str]) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        input_sources = sum(bool(source) for source in (args.state, args.tokens, args.scramble is not None))
        if input_sources > 1:
            raise PuzzleError("Use only one input style: --state, positional tokens, or --scramble.")

        if args.scramble is not None:
            if args.scramble < 0:
                raise PuzzleError("--scramble must be zero or greater.")
            start = random_scramble(args.scramble, args.seed)
        elif args.state:
            start = parse_state_text(args.state)
        elif args.tokens:
            start = parse_state(args.tokens)
        else:
            print("No state given, so running the tiny example from the prompt.\n")
            start = parse_state_text("11 12 13 14 21 22 23 24 31 32 33 34 41 42 00 43 44")

        moves, stats = solve(start, max_seconds=args.max_seconds, algorithm=args.algorithm)
        print_solution(start, moves, stats, args.show_states)
        return 0
    except (PuzzleError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
