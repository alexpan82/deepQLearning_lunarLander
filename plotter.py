# Takes metadata.npy outputted from training
# Plots
import matplotlib.pyplot as plt
import numpy as np
import argparse

def rolling_avg(arr, window_size):
    avg_list = []
    arr_size = len(arr)
    for i in range(0, arr_size-window_size):
        avg_list.append(sum(arr[i:i+window_size])/window_size)
    avg_list = np.array(avg_list)
    avg_list = np.pad(avg_list, (window_size, 0), mode='constant', constant_values=(np.nan,))
    return avg_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for anime_face_track')
    parser.add_argument('--output', required=True,
                        metavar="<output>",
                        help='Name of output png file')
    parser.add_argument('--npy', required=True,
                        metavar='<files>', nargs='*',
                        help='Any number of metadata.npy files (from main.py) to plot')
    parser.add_argument('--legend', required=False,
                        metavar='<legend>', default=-1, type=int,
                        help='''What to plot in legend. Type integer index corresponding to one of:
                            [frame_counter, epsilon_frame, replay_size, alpha,
                            nn_update_freq, batch_size]''')
    parser.add_argument('--avg', required=False,
                        metavar='<avg>', default=-1, type=int,
                        help='''Calculate rolling average of reward over window size <avg>''')

    args = parser.parse_args()
    print(args)

    npy_dic = {}  # hold episodes of rewards
    legend_dic = {}  # hold legend

    x_values = []
    for npy in args.npy:
        arr = np.load(npy, allow_pickle=True)
        episodes = arr[0]
        if len(x_values) < len(episodes):
            x_values = episodes
        reward_values = arr[2]
        q_values = arr[1]
        npy_dic[npy] = (reward_values, q_values,)

        # Make legend
        parameters = arr[-1]
        print(npy, parameters)
        if args.legend == -1:
            legend_dic[npy] = npy
        else:
            legend_dic[npy] = parameters[args.legend]

    # Pad y-values with nans then plot
    fig, axs = plt.subplots(2)
    if args.avg > -1:
        axs[0].set_title("Total Reward (Rolling Average)")
    else:
        axs[0].set_title("Total Reward")
    axs[1].set_title("Avg Q value for a state")
    fig.tight_layout()

    max_len = len(x_values)
    for npy in npy_dic.keys():
        # y_values = npy_dic[npy]
        reward_values = npy_dic[npy][0]
        q_values = npy_dic[npy][1]
        pad_len = max_len - len(reward_values)
        print("Number Episodes: ", len(reward_values))

        if args.avg > -1:
            reward_values = rolling_avg(reward_values, args.avg)

        reward_values = np.pad(reward_values, (0, pad_len), mode='constant', constant_values=(np.nan,))
        q_values = np.pad(q_values, (0, pad_len), mode='constant', constant_values=(np.nan,))

        axs[0].plot(x_values, reward_values, label=round(legend_dic[npy]))
        axs[1].plot(x_values, q_values, label=round(legend_dic[npy]))

    plt.legend()
    plt.show()
    fig.savefig(args.output)
    plt.close()
