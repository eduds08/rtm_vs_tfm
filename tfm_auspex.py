import os
import matplotlib.pyplot as plt
import numpy as np
from framework import file_m2k, pre_proc, post_proc
from framework.data_types import ImagingROI
from imaging import tfm

'''
folder,
speed (m/s),
corner_roi[0] (mm),
size (mm)
'''
panther_tests = {
    'teste1': [
        './arquivos_m2k/teste1_longe_fio_a_direita.m2k',
        1500.,
        [8, 0, 31],
        [6, 7],
    ],
    'teste2': [
        './arquivos_m2k/teste2_perto_fio_a_esquerda.m2k',
        1500.,
        [-16, 0, 6],
        [11, 10],
    ],
    'teste3': [
        './arquivos_m2k/teste3_aço_inclinado.m2k',
        1500.,
        [-30, 0, 5],
        [60, 17],
    ],
    'teste4': [
        './arquivos_m2k/teste4_aluminio_com_furo.m2k',
        6420.,
        [-6, 0, 16],
        [10, 8],
    ],
    'teste5': [
        './arquivos_m2k/teste5_aluminio_com_furo_vertical.m2k',
        6420.,
        [2, 0, 36],
        [8, 8],
    ],
    'teste6': [
        './arquivos_m2k/teste6_aluminio_semi_circulo.m2k',
        6420.,
        [-85, 0, 0],
        [170, 90],
    ],
    'teste7': [
        './arquivos_m2k/2024-08-21 Oblongos.m2k',
        6420.,
        [-35, 0, 10],
        [65, 50],
    ],
}

if not os.path.exists('tfm_auspex_pngs'):
    os.makedirs('tfm_auspex_pngs')

selected_test_name = 'teste1'

selected_test = panther_tests[selected_test_name]

file = selected_test[0]
data = file_m2k.read(file, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

L = data.ascan_data.shape[-1]
_ = pre_proc.hilbert_transforms(data, np.arange(L))

corner_roi = np.zeros((1, 3))
corner_roi[0] = selected_test[2]
size = np.array(selected_test[3])

roi = ImagingROI(corner_roi, height=size[1], width=size[0], h_len=int(10*size[1]), w_len=int(10*size[0]), depth=1.0, d_len=1)

c = selected_test[1]

chave = tfm.tfm_kernel(data, roi=roi, c=c)
result = post_proc.normalize(post_proc.envelope(data.imaging_results[chave].image, -2))

np.save(f'./tfm_npys/tfm_{selected_test_name}.npy', result)

print((roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]))

plt.figure()
plt.imshow(result, aspect='auto', extent=(roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]))
plt.title(f'TFM {roi.w_len}x{roi.h_len} [px] - {size[0]}x{size[1]} [mm] - dz: {size[0] / roi.w_len}mm | dx: {size[1] / roi.h_len}mm')
plt.grid()
plt.savefig(f'./tfm_auspex_pngs/tfm_{selected_test_name}.png')
plt.close()
