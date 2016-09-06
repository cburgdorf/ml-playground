import numpy as np

def gen_data():
    arr = np.array([1,0,0]).repeat(12);
    np.random.shuffle(arr)
    return arr

def reshape(arr):
    return arr.reshape(6,6)

def mark_line(arr):
    print (arr[0][0])

def get_up_left(x, y):
    return (x - 1, y -1)

def get_up (x, y):
    return (x - 1, y)

def get_up_right (x, y):
    return (x - 1, y + 1)

def get_right (x, y):
    return (x, y + 1)

def get_down_right (x, y):
    return (x + 1, y + 1)

def get_down (x, y):
    return (x + 1, y)

def get_down_left (x, y):
    return (x + 1, y -1)

def get_left (x, y):
    return (x, y -1)

def is_valid_cord (x):
    return x >= 0 and x <= 5

def is_positive(grid, x, y):
    if not is_valid_cord(x) or not is_valid_cord(y):
        return False
    return grid[x][y] == 1


def is_beginning_of_straight_line(grid, x,y, navFn):
    for i in range(0, 4):
        if not is_positive(grid, x, y):
            return False
        x, y = navFn(x, y)

    return True

def draw_line(grid, x,y, navFn):
    for i in range(0, 4):
        grid[x, y] = 1
        x, y = navFn(x, y)

def has_straight_line(grid, xDim, yDim):
    for x in range(0, xDim):
        for y in range(0, yDim):
            if is_beginning_of_straight_line(grid, x, y, get_up_left):
                print("has upper left straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_up):
                print("has upper straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_up_right):
                print("has upper right straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_right):
                print("has right straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_down_right):
                print("has down right straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_down):
                print("has down line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_down_left):
                print("has down left straight line: x:" + str(x) + " y:" + str(y))
            if is_beginning_of_straight_line(grid, x, y, get_left):
                print("has left straight line: x:" + str(x) + " y:" + str(y))

def mark_straight_lines(grid, xDim, yDim):
    copy_grid = reshape(np.array([0]).repeat(36));
    for x in range(0, xDim):
        for y in range(0, yDim):
            if is_beginning_of_straight_line(grid, x, y, get_up_left):
                draw_line(copy_grid, x, y, get_up_left)
            if is_beginning_of_straight_line(grid, x, y, get_up):
                draw_line(copy_grid, x, y, get_up)
            if is_beginning_of_straight_line(grid, x, y, get_up_right):
                draw_line(copy_grid, x, y, get_up_right)
            if is_beginning_of_straight_line(grid, x, y, get_right):
                draw_line(copy_grid, x, y, get_right)
            if is_beginning_of_straight_line(grid, x, y, get_down_right):
                draw_line(copy_grid, x, y, get_down_right)
            if is_beginning_of_straight_line(grid, x, y, get_down):
                draw_line(copy_grid, x, y, get_down)
            if is_beginning_of_straight_line(grid, x, y, get_down_left):
                draw_line(copy_grid, x, y, get_down_left)
            if is_beginning_of_straight_line(grid, x, y, get_left):
                draw_line(copy_grid, x, y, get_left)
    return copy_grid

def test ():
    total_lines = 0
    total_positives = 0

    for x in range(0, 1000):
        grid = reshape(gen_data())
        print("\ngenerated grid:\n")
        print(grid.reshape(1, 6, 6))
        marked_grid = mark_straight_lines(grid, 6, 6)
        num_lines = marked_grid.sum() / 4
        total_lines = total_lines + num_lines
        if (num_lines > 0):
            total_positives = total_positives + 1
        print("\nmarked lines " + str(num_lines)  + ":\n")
        print(marked_grid)

    print ("\nTotal Positives: " + str(total_positives))
    print ("Total Lines: " + str(total_lines))

def generate_samples(sample_count):
    total_lines = 0
    total_positives = 0

    training_data = None
    target_data = None
    target_data_simple = None;
    for x in range(0, sample_count):
        grid = reshape(gen_data())
        if training_data == None:
            training_data = grid.reshape(1, 6, 6)
        else:
            training_data = np.concatenate([training_data, grid.reshape(1, 6, 6)])
        marked_grid = mark_straight_lines(grid, 6, 6)
        num_lines = marked_grid.sum() / 4
        total_lines = total_lines + num_lines
        if (num_lines > 0):
            total_positives = total_positives + 1
        if target_data == None:
            target_data = marked_grid.reshape(1, 6, 6)
            target_data_simple = np.array([ 1 if num_lines > 0 else 0 ])
        else:
            target_data = np.concatenate([target_data, marked_grid.reshape(1, 6, 6)])
            target_data_simple = np.concatenate([target_data_simple, np.array([1 if num_lines > 0 else 0])])

    print("\nTotal Positives: " + str(total_positives))
    print("Total Lines: " + str(total_lines))

    return (training_data.reshape(sample_count, 1, 6 , 6),
            target_data.reshape(sample_count, 1, 6, 6),
            target_data_simple.flatten())


print generate_samples(3)
