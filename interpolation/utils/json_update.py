import matplotlib.pyplot as plt
import numpy as np

def visualize_ramp_fix(input_vals, output_vals,savepath='ramp.png'):
    plt.figure(figsize=(12, 4))

    # Plot input
    plt.subplot(1, 2, 1)
    plt.plot(input_vals, marker='o')
    plt.title("Input Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    # Plot output
    plt.subplot(1, 2, 2)
    plt.plot(output_vals, marker='o', color='green')
    plt.title("Output After Ramp Fix")
    plt.xlabel("Index")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(savepath)
# def fix_ramps(vals):
#     vals = vals.copy()
#     n = len(vals)
#     i = 0

#     while i < n:
#         # --- Skip leading 0s/1s ---
#         while i < n and vals[i] in [0, 1]:
#             i += 1

#         # --- Up Ramp ---
#         up_start = i
#         while i < n and 0 < vals[i] < 1:
#             i += 1
#         up_end = i

#         if up_start < up_end and i < n and vals[i] == 1:
#             ramp = vals[up_start:up_end]
#             ramp_len = len(ramp)
#             insert_start = up_start - ramp_len
#             if insert_start >= 0:
#                 for j in range(ramp_len):
#                     vals[insert_start + j] = ramp[j]
#             for j in range(up_start, up_end):
#                 vals[j] = 1

#         # --- 1s block ---
#         while i < n and vals[i] == 1:
#             i += 1
#         one_end = i

#         # --- Down Ramp ---
#         down_start = i
#         while i < n and 0 < vals[i] < 1:
#             i += 1
#         down_end = i

#         if down_start < down_end and down_end < n and vals[down_end] == 0:
#             ramp = vals[down_start:down_end]
#             ramp_len = len(ramp)
#             insert_end = down_end + ramp_len
#             if insert_end <= n:
#                 for j in range(ramp_len):
#                     vals[down_end + j] = ramp[j]
#             for j in range(down_start, down_end):
#                 vals[j] = 1

#         i = down_end + 1

#     return vals
def fix_ramps(vals):
    vals = vals.copy()
    n = len(vals)
    i = 0

    while i < n:
        # Skip 0s and 1s
        while i < n and vals[i] in [0, 1]:
            i += 1

        # --- Up Ramp ---
        up_start = i
        while i < n and 0 < vals[i] < 1:
            i += 1
        up_end = i

        if up_start < up_end and i < n and vals[i] == 1:
            ramp_len = up_end - up_start
            insert_start = up_start - ramp_len
            if insert_start >= 0:
                for j in range(ramp_len):
                    vals[insert_start + j] = vals[up_start + j]
            for j in range(up_start, up_end):
                vals[j] = 1
            i += 1  # move past the 1

        # Skip consecutive 1s
        while i < n and vals[i] == 1:
            i += 1

        # --- Down Ramp ---
        down_start = i
        while i < n and 0 < vals[i] < 1:
            i += 1
        down_end = i

        if down_start < down_end and down_end < n and vals[down_end] == 0:
            ramp_len = down_end - down_start
            insert_end = down_end + ramp_len
            if insert_end <= n:
                for j in range(ramp_len):
                    vals[down_end + j] = vals[down_start + j]
            for j in range(down_start, down_end):
                vals[j] = 1
            i = down_end + ramp_len
        else:
            break

    return vals

def fix_ramps_np(values):
    vals = np.array(values, dtype=np.float32)
    n = len(vals)
    i = 0

    while i < n:
        # Skip 0s and 1s
        while i < n and vals[i] in (0.0, 1.0):
            i += 1

        # --- Up Ramp ---
        up_start = i
        while i < n and 0.0 < vals[i] < 1.0:
            i += 1
        up_end = i

        if up_start < up_end and i < n and vals[i] == 1.0:
            ramp_len = up_end - up_start
            insert_start = up_start - ramp_len
            if insert_start >= 0:
                vals[insert_start:up_start] = vals[up_start:up_end]
            vals[up_start:up_end] = 1.0
            i += 1  # move past 1

        # Skip 1s
        while i < n and vals[i] == 1.0:
            i += 1

        # --- Down Ramp ---
        down_start = i
        while i < n and 0.0 < vals[i] < 1.0:
            i += 1
        down_end = i

        if down_start < down_end and down_end < n and vals[down_end] == 0.0:
            ramp_len = down_end - down_start
            insert_end = down_end + ramp_len
            if insert_end <= n:
                vals[down_end:down_end + ramp_len] = vals[down_start:down_end]
            vals[down_start:down_end] = 1.0
            i = down_end + ramp_len
        else:
            break

    return vals.tolist()
# Example usage
if __name__ == "__main__":
    data = [0,0,0,0,0,0,0,0,0.1,0.4,0.7,1,1,1,1,1,0.7,0.4,0.1,0,0,0,0,0,0,0,0,0.2,0.4,1,1,1,1,0.4,0.2,0,0,0,0]
    result = fix_ramps_np(data.copy())
    visualize_ramp_fix(data, result)
    print(result)