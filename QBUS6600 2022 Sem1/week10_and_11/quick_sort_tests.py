import numpy as np
import time


def quick_sort(x):
    if len(x) == 1:
        return x
    elif len(x) == 0:
        return []

    pivot = x.pop()
    
    left_of_pivot = []
    right_of_pivot = []
    for y in x:
        if y < pivot:
            left_of_pivot.append(y)
        else:
            right_of_pivot.append(y)
            
    return quick_sort(left_of_pivot) + [pivot] + quick_sort(right_of_pivot)


if __name__ == "__main__":
    for i in range(5):
        np.random.seed(0)
        x = list(np.random.rand(1_000_000))
    
        start = time.time()
        quick_sort(x)
        end = time.time()
        
        print(f"Trial {i + 1} of 5: Time Taken {round(end - start, 2)} seconds")
    