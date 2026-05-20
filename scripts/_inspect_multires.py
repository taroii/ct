import pickle
with open('cache/multiresolution_results.pkl', 'rb') as f:
    r = pickle.load(f)
for res in (512, 256, 128):
    d = r[res]
    print(f'{res}^2:')
    for it in (50, 100, 200, 300, 500):
        rs = d['ierrs_single'][it-1]
        rt = d['ierrs_two'][it-1]
        gap = (rs - rt)/rs*100
        print(f'  iter {it:3d}: single={rs:.4f}  two={rt:.4f}  gap={gap:+.1f}%')
    print()
