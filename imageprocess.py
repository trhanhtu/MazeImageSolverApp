from typing import List, Tuple
import cv2 
import numpy as np
import imutils

from mazeenum import Maze

def GetImageForEachStep(originalImg:np.ndarray) -> List[np.ndarray]:
    try:
        image_list: List[np.ndarray] = [originalImg]
        convert_to_gray(image_list)
        filter_image(image_list)
        find_edge(image_list)
        contours = grab_contours(image_list)
        location = crop_maze(image_list, contours)
        array_of_images = split_boxes(image_list[-1], 10)
        maze,start,end = convert_arr_img_to_number(array_of_images)
        img_path = DFS(image_list, maze,start,end)
        image_list.append(img_path)
        inv = get_inv_perspective(image_list[0], img_path, location)
        image_list.append(inv)
        combined = cv2.addWeighted(image_list[0], 0.7, inv, 1, 0)
        image_list.append(combined)
    finally:
        return image_list

def convert_to_gray(image_list: List[np.ndarray]) -> None:
    gray = cv2.cvtColor(image_list[0], cv2.COLOR_BGR2GRAY)
    image_list.append(gray)

def filter_image(image_list: List[np.ndarray]) -> None:
    # bảo tồn góc cạnh tốt
    bfilter = cv2.bilateralFilter(image_list[1], 13, 20, 20)
    image_list.append(bfilter)

def find_edge(image_list: List[np.ndarray]) -> None:
    edged = cv2.Canny(image_list[2], 30, 180)
    image_list.append(edged)

def grab_contours(image_list: List[np.ndarray]) -> List[np.ndarray]:
    blue_mask = find_blue_regions(image_list[0])
    # Find contours in the blue mask
    keypoints = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    new_img = cv2.drawContours(image_list[0].copy(), contours, -1, (0, 255, 0), 3)
    image_list.append(new_img)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    return contours

def find_blue_regions(image: np.ndarray) -> np.ndarray:
    # Define the lower and upper bounds for blue color in HSV color space
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

def crop_maze(image_list: List[np.ndarray], contours: List[np.ndarray]) -> np.ndarray:
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(image_list[0], location)
    image_list.append(result)
    return location

def get_perspective(img: np.ndarray, location: np.ndarray, height: int = 390, width: int = 390) -> np.ndarray:
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def split_boxes(board: np.ndarray, img_size: int) -> np.ndarray:
    rows: List[np.ndarray] = np.vsplit(board, Maze.HEIGHT)
    boxes: np.ndarray = np.empty((Maze.WIDTH, Maze.HEIGHT, img_size, img_size, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        cols: List[np.ndarray] = np.hsplit(r, Maze.WIDTH)
        for j, box in enumerate(cols):
            boxes[i, j] = cv2.resize(box, (img_size, img_size))
    return boxes

def convert_arr_img_to_number(arr_img: np.ndarray) -> tuple[np.ndarray,tuple[int,int],tuple[int,int]]:
    start = None
    end = None
    maze: np.ndarray = np.empty((Maze.WIDTH, Maze.HEIGHT), dtype=np.uint8)
    for i in range(Maze.HEIGHT):
        for j in range(Maze.WIDTH):
            maze[i, j] = read_color(arr_img[i, j])
            if(maze[i,j] == Maze.START):
                start = (i,j)
            elif(maze[i,j] == Maze.END):
                end = (i,j)
            cv2.imwrite(f"../pic/{i}_{j}_{maze[i,j]}.png",arr_img[i,j])
    print(start," ",end)
    return maze ,start , end

def read_color(picture: np.ndarray) -> int:
    hsv_image = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'white': ((0, 0, 200), (180, 50, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'red': ((0, 100, 100), (10, 255, 255)),
        'orange': ((10, 100, 100), (40, 255, 255)),
        'green': ((25, 50, 50), (80, 255, 255)),
        'blue': ((90, 50, 50), (120, 255, 255))
    }
    color_counts = {color: 0 for color in color_ranges}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        color_counts[color_name] = cv2.countNonZero(mask)
    predominant_color = max(color_counts, key=color_counts.get)
    if color_counts[predominant_color] > 20:
        return {
            'white': Maze.EMPTY,
            'yellow': Maze.PATH,
            'red': Maze.START,
            'orange': Maze.START,
            'green': Maze.END,
            'blue': Maze.WALL
        }[predominant_color]
    else:
        return Maze.WALL

def DFS(image_list: List[np.ndarray], matrix: np.ndarray, start:Tuple[int, int] , end:Tuple[int, int]) -> np.ndarray:

    if start is None or end is None:
        print("Không tìm thấy điểm bắt đầu/kết thúc")
        return
    visited = [[False for _ in range(Maze.HEIGHT)] for _ in range(Maze.WIDTH)]
    path = find_path(matrix, start, end, visited)
    img_path = draw_path_only(image_list[-1], path)
    return img_path

def find_path(matrix: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], visited: List[List[bool]]) -> List[Tuple[int, int]]:
    if start == end:
        return [end]
    visited[start[0]][start[1]] = True
    for neighbor in get_neighbors(matrix, start):
        if not visited[neighbor[0]][neighbor[1]]:
            path = find_path(matrix, neighbor, end, visited)
            if path:
                return [start] + path
    return None

def get_neighbors(matrix: np.ndarray, point: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = point
    neighbors = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] != 5:
            neighbors.append((nx, ny))
    return neighbors

def draw_path_only(img: np.ndarray, path: List[Tuple[int, int]]) -> np.ndarray:
    blank = np.zeros_like(img)
    orange = (0, 165, 255)
    cell_size = 10
    if path:
        for point in path:
            y, x = point
            center_x = (x * cell_size) + cell_size // 2
            center_y = (y * cell_size) + cell_size // 2
            cv2.circle(blank, (center_x, center_y), 3, orange, -1)
    return blank

def get_inv_perspective(img: np.ndarray, masked_num: np.ndarray, location: np.ndarray, height: int = 390, width: int = 390) -> np.ndarray:
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result