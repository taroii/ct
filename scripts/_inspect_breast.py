import pickle
with open('cache/ct2_breast_recon.pkl', 'rb') as f:
    r = pickle.load(f)
print('keys:', list(r.keys())[:15])
print('snapshot iters:', sorted(r['snapshots_single'].keys()))
print()
print('ct-2 breast RMSEs:')
for it in (20, 50, 100, 200, 300, 500):
    rs = r['ierrs_single'][it-1]
    rt = r['ierrs_two'][it-1]
    gap = (rs - rt) / rs * 100
    print(f'  iter {it:3d}: single={rs:.4f}  two={rt:.4f}  gap={gap:+.1f}%')
