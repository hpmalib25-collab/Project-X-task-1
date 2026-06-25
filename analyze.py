import cv2
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt

MODULO = 10**9 + 7

GRID_SIZE = 40
CELL_SIZE = 20

PIXEL_BRIGHT = 200
LIGHT_THRESHOLD = 150


def extract_grid(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y = row * CELL_SIZE
            x = col * CELL_SIZE

            cell = gray[y:y+CELL_SIZE, x:x+CELL_SIZE]

            light_count = np.sum(cell > PIXEL_BRIGHT)

            grid[row, col] = light_count >= LIGHT_THRESHOLD

    return grid


def bfs_with_path(grid):
    rows, cols = grid.shape

    start = (0, 0)
    goal = (rows - 1, cols - 1)

    if not grid[start] or not grid[goal]:
        return None, None

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start] = True

    parent = {}

    q = deque([(0, 0)])

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while q:
        r, c = q.popleft()

        if (r, c) == goal:

            path = []
            cur = goal

            while cur != start:
                path.append(cur)
                cur = parent[cur]

            path.append(start)
            path.reverse()

            return len(path), path

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            if (
                0 <= nr < rows and
                0 <= nc < cols and
                grid[nr, nc]
                and not visited[nr, nc]
            ):
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    return None, None

def draw_path(image_path, path_cells, output_path):
    img = cv2.imread(image_path)

    if path_cells is None:
        cv2.imwrite(output_path, img)
        return

    points = []

    for r, c in path_cells:
        x = c * CELL_SIZE + CELL_SIZE // 2
        y = r * CELL_SIZE + CELL_SIZE // 2
        points.append((x, y))

    # Draw red shortest path
    for i in range(len(points) - 1):
        cv2.line(
            img,
            points[i],
            points[i + 1],
            (0, 0, 255),
            2
        )

    # Green start
    cv2.circle(img, points[0], 6, (0, 255, 0), -1)

    # Blue goal
    cv2.circle(img, points[-1], 6, (255, 0, 0), -1)

    cv2.imwrite(output_path, img)



def solve_all_mazes(maze_dir):
    os.makedirs("solved", exist_ok=True)
    files = sorted(
        f for f in os.listdir(maze_dir)
        if f.lower().endswith(".png")
    )

    print(f"Found {len(files)} maze images")

    product = 1
    solvable = 0

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(maze_dir, fname)

        try:
            grid = extract_grid(path)
            length, path_cells = bfs_with_path(grid)
            
            output_file = os.path.join("solved", fname)

            draw_path(
                path,
                path_cells,
                output_file
            )

            if length is None:
                print(f"[{idx}] {fname} -> UNSOLVABLE")
                continue

            product = (product * length) % MODULO
            solvable += 1

            print(f"[{idx}] {fname} -> Path Length = {length}")

        except Exception as e:
            print(f"[{idx}] {fname} -> ERROR: {e}")

    print("\n==============================")
    print("Solvable mazes:", solvable)
    print("Final Product mod (1e9+7):")
    print(product)

    return product


def debug_first_maze(maze_dir):
    files = sorted(
        f for f in os.listdir(maze_dir)
        if f.lower().endswith(".png")
    )

    if not files:
        print("No images found")
        return

    fname = files[0]

    path = os.path.join(maze_dir, fname)

    grid = extract_grid(path)

    print("Showing:", fname)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray")
    plt.title(fname)
    plt.axis("off")
    plt.show()


def main():
    maze_dir = "."

    files = sorted(
        f for f in os.listdir(maze_dir)
        if f.lower().endswith(".png")
    )

    MOD = 10**9 + 7
    product = 1

    solvable = 0
    unsolvable = 0

    for fname in files:

        grid = extract_grid(fname)

        length, path = bfs_with_path(grid)

        if length is None:
            print(f"{fname}: UNSOLVABLE")
            unsolvable += 1
            continue

        print(f"{fname}: {length}")

        product = (product * length) % MOD
        solvable += 1

    print(
    f"{fname}: UNSOLVABLE | "
    f"start={grid[0,0]} end={grid[39,39]}"
)
    print("\n" + "=" * 40)
    print("Solvable mazes  :", solvable)
    print("Unsolvable mazes:", unsolvable)
    print("Final Product Modulo 1e9+7:")
    print(product)

if __name__ == "__main__":
    main()