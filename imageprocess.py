import traceback
from typing import List, Tuple
import cv2 
import numpy as np
import imutils
from mazeenum import Maze
from workspace import Workspace

def GetImageForEachStep(originalImg:np.ndarray,workspace: Workspace) -> List[np.ndarray]:
    image_list = [originalImg]
    
    for i in range(4):
        try:
            resultCode:int = DoStep(image_list,i,workspace)

            if( resultCode != -1):
                break

            workspace.logging("[-] auto increase algorithm complex")
            del image_list[1:]
            
        except Exception:
            traceback.print_exc()
            workspace.logging("[-] auto increase algorithm by exception ")

    workspace.logging( f"----------------\n[+] total:{len(image_list)}")
    return image_list

def DoStep(image_list:List[np.ndarray],indexLoop: int,workspace: Workspace)-> int:
    originalImg = image_list[0]
    grayImg,filterImg = preprocess_image(originalImg,indexLoop)
    edgedImg = find_edge(filterImg)
    contours,drawImg = grab_contours(edgedImg,originalImg)
    location,cropImg = crop_maze(originalImg, contours)

    image_list.extend([ grayImg, filterImg, edgedImg, drawImg])
    if(cropImg is None):
        return -1
    
    array_of_images = split_boxes(cropImg, 10)
    maze,start,end = convert_arr_img_to_number(array_of_images,workspace)
    img_path = DFS(cropImg, maze,start,end,workspace)

    image_list.append(cropImg)
    if(img_path is None):
        return -1
    
    mazeWithPath = cv2.addWeighted(cropImg, 0.7, img_path, 1, 0)
    invMazePath = get_inv_perspective(originalImg, mazeWithPath, location)
    inv = get_inv_perspective(originalImg, img_path, location)
    combined = cv2.addWeighted(originalImg, 0.7, inv, 1, 0)
    
    image_list.extend([mazeWithPath,invMazePath,combined])

    return 0

def preprocess_image(image: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
    preimage = image
    if(i == 1 or i== 3):
        preimage = increase_brightness_log(image)

    gray = convert_to_gray(preimage)
    if( i == 0):
        filterImg = filter_image(gray)
        return gray , filterImg
    
    blue_regions = find_blue_regions(preimage)
    return gray, blue_regions

def increase_brightness_log(image, factor=25.5):
    # Convert image to floating point
    image_float = image.astype(float)
    
    # Add a small constant value to avoid zero values
    scaled_image = image_float + 1.0
    
    # Apply the logarithmic transformation
    log_transformed = (np.log(scaled_image) / np.log(256)) * factor
    
    # Scale back to the range [0, 255] and convert to unsigned integer
    result = np.uint8(cv2.normalize(log_transformed, None, 0, 255, cv2.NORM_MINMAX))
    
    return result


def convert_to_gray(image:np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def filter_image(image:np.ndarray) -> np.ndarray:
    # bảo tồn góc cạnh tốt
    bfilter = cv2.bilateralFilter(image, 13, 20, 20)
    return bfilter

def find_edge(image_filted:np.ndarray) -> np.ndarray:
    edged = cv2.Canny(image_filted, 30, 180)
    return edged

def grab_contours(imageEdge:np.ndarray,originalImg:np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    
    # Find contours in the blue mask
    keypoints = cv2.findContours(imageEdge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    new_img = cv2.drawContours(originalImg.copy(), contours, -1, (0, 255, 0), 3)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    return contours , new_img

def find_blue_regions(image: np.ndarray) -> np.ndarray:
    # Define the lower and upper bounds for blue color in HSV color space
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

def crop_maze(originalImage:np.ndarray, contours: List[np.ndarray]) -> Tuple[np.ndarray | None,np.ndarray | None]:
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(originalImage, location)
    return location , result

def get_perspective(img: np.ndarray, location: np.ndarray | None, height: int = 390, width: int = 390) -> np.ndarray | None:
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

def convert_arr_img_to_number(arr_img: np.ndarray,workspace:Workspace) -> tuple[np.ndarray, tuple[int, int] | None, tuple[int, int] | None]:
    start = None
    end = None
    maze: np.ndarray = np.empty((Maze.WIDTH, Maze.HEIGHT), dtype=np.uint8)
    for i in range(Maze.HEIGHT):
        for j in range(Maze.WIDTH):
            maze[i, j] = read_color(arr_img[i, j],workspace)
            if(maze[i,j] == Maze.START):
                start = (i,j)
            elif(maze[i,j] == Maze.END):
                end = (i,j)
            # cv2.imwrite(f"../pic/{i}_{j}_c{maze[i,j]}.png",arr_img[i,j])
    return maze ,start , end

def read_color(picture: np.ndarray,workspace:Workspace) -> int:
    hsv_image = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'white': ((0, 0, 200), (180, 150, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'red': ((0, 100, 100), (10, 255, 255)),
        'orange': ((10, 100, 100), (40, 255, 255)),
        'brown': ((10, 50, 50), (30, 255, 150)),
        'green': ((25, 50, 50), (80, 255, 255)),
        'blue': ((90, 50, 50), (120, 255, 255)),
        'gray': ((0, 0, 0), (180, 50, 255))

    }
    color_counts = {color: 0 for color in color_ranges}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        color_counts[color_name] = cv2.countNonZero(mask)
    predominant_color = max(color_counts, key=color_counts.get)
    if predominant_color not in color_ranges:
        workspace.logging(f"[?] Unrecognized color detected: {predominant_color}")
        return Maze.WALL
    
    if color_counts[predominant_color] > 20:
        return {
            'white': Maze.EMPTY,
            'gray': Maze.EMPTY,
            'yellow': Maze.PATH,
            'red': Maze.START,
            'orange': Maze.START,
            'brown' : Maze.START,
            'green': Maze.END,
            'blue': Maze.WALL,
        }[predominant_color]
    
    workspace.logging(f"[?] Max color is too small: {predominant_color}")
    return Maze.WALL

def DFS(imageCrop: np.ndarray, matrix: np.ndarray, start:Tuple[int, int] |None, end:Tuple[int, int]|None,workspace: Workspace) -> np.ndarray | None:

    if start is None or end is None:
        workspace.logging(f"[?] missing start= {start} /end= {end}")
        return None
    visited = [[False for _ in range(Maze.HEIGHT)] for _ in range(Maze.WIDTH)]
    path = find_path(matrix, start, end, visited)
    if(path is None or len(path) == 0):
        workspace.logging("[?] something block the way")
    else:
        workspace.logging(f"[+] found path from {start} to {end}")
    img_path = draw_path_only(imageCrop, path)
    return img_path

def find_path(matrix: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], visited: List[List[bool]]) -> List[Tuple[int, int]] | None:
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

def draw_path_only(img: np.ndarray, path: List[Tuple[int, int]] | None) -> np.ndarray:
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

def get_inv_perspective(img: np.ndarray, masked_num: np.ndarray, location: np.ndarray | None, height: int = 390, width: int = 390) -> np.ndarray:
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result