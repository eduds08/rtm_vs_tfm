import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def calculate_api(data, dz, dx):
    pixel_size = dz * dx
    half_max_value = np.amax(data) / 2
    pixels_len = np.sum(data >= half_max_value)

    api_result = pixels_len * pixel_size

    return api_result


def plot_rtm(image, selected_test, selected_test_name, api_rtm):
    # Receptor's position
    number_of_receptors = 64
    receptor_z = []
    for rp in range(number_of_receptors):
        receptor_z.append((selected_test[4] * rp) / selected_test[1][2])
    receptor_z = (np.int32(np.asarray(receptor_z)) + np.int32((selected_test[1][0] - receptor_z[-1]) / 2))

    x_values = (np.arange(selected_test[1][0]) - receptor_z[31]) * selected_test[1][2] * 1e3
    y_values = np.arange(selected_test[1][1]) * selected_test[1][3] * 1e3

    plt.figure()
    plt.imshow(image, aspect='auto', extent=(x_values[0], x_values[-1], y_values[-1], y_values[0]))

    plt.xlim(selected_test[2][0], selected_test[2][0] + selected_test[3][0])
    plt.ylim(selected_test[2][2] + selected_test[3][1], selected_test[2][2])

    if api_rtm is not None:
        plt.title(f'RTM - API: {api_rtm:.2f}')
    else:
        plt.title(f'RTM')
    plt.grid()
    plt.savefig(f'./rtm_api_pngs/rtm_{selected_test_name}.png')
    plt.close()
    # plt.show()


def plot_tfm(image, selected_test, api_tfm):
    plt.figure()
    plt.imshow(image, aspect='auto', extent=(selected_test[2][0], selected_test[2][0] + selected_test[3][0], selected_test[2][2] + selected_test[3][1], selected_test[2][2]))

    if api_tfm is not None:
        plt.title(f'TFM - API: {api_tfm:.2f}')
    else:
        plt.title(f'TFM')
    plt.grid()
    plt.savefig(f'./tfm_api_pngs/tfm_{selected_test_name}.png')
    plt.close()
    # plt.show()


'''
speed (m/s),
(grid_size_z, grid_size_x, dz, dx),
corner_roi[0] (mm),
size (mm)
'''
panther_tests = {
    'teste1': [
        1500.,
        (2100, 2100, 3.0e-5, 3.0e-5),
        [8, 0, 31],
        [6, 7],
        0.6e-3,
    ],
    'teste2': [
        1500.,
        (1750, 2100, 2.5e-5, 3.0e-5),
        [-16, 0, 6],
        [11, 10],
        0.6e-3,
    ],
    'teste3': [
        1500.,
        (2000, 2000, 6e-5, 6e-5),
        [-30, 0, 5],
        [60, 17],
        0.6e-3,
    ],
    'teste4': [
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
        [-6, 0, 16],
        [10, 8],
        0.6e-3,
    ],
    'teste5': [
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
        [2, 0, 36],
        [8, 8],
        0.6e-3,
    ],
    'teste6': [
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
        [-85, 0, 0],
        [170, 90],
        0.6e-3,
    ],
    'teste7': [
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
        [-35, 0, 10],
        [65, 50],
        0.5e-3,
    ],
}

testes = ['teste1', 'teste2', 'teste3', 'teste4', 'teste5', 'teste6', 'teste7']

if not os.path.exists('tfm_api_pngs'):
    os.makedirs('tfm_api_pngs')

if not os.path.exists('rtm_api_pngs'):
    os.makedirs('rtm_api_pngs')

for teste in testes:
    selected_test_name = teste

    selected_test = panther_tests[selected_test_name]

    tfm = np.load(f'./tfm_npys/tfm_{selected_test_name}.npy')
    rtm_fmc = np.load(f'./rtm_npys/{selected_test_name}.npy')
    envelope_rtm = np.abs(hilbert(rtm_fmc, axis=0))

    api_tfm = None
    api_rtm = None
    if selected_test_name in ['teste1', 'teste2', 'teste4', 'teste5']:
        api_tfm = calculate_api(tfm, 0.1, 0.1)
        api_rtm = calculate_api(envelope_rtm, selected_test[1][2] * 1e3, selected_test[1][3] * 1e3)

        with open(r'./api_values.txt', 'a') as file:
            file.write(f'''Teste {selected_test_name[-1]}:
API TFM - {api_tfm}
API RTM - {api_rtm}\n\n''')

    plot_rtm(envelope_rtm, selected_test, selected_test_name, api_rtm)
    plot_tfm(tfm, selected_test, api_tfm)
