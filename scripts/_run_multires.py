"""Drive compare_methods_multiresolution.run_reconstruction_for_mfact across
all three resolutions at itermax=500 (paper config) and dump a fresh cache
for the persistence plot. Run before presentation_persistence_plot.py."""
import pickle
import sys
sys.path.insert(0, "scripts")
import compare_methods_multiresolution as cm

cm.cutoffparm = 4.0
cm.cutoffparm_lo = 8.0
cm.eps_hi = cm.eps
cm.eps_lo = 1.25 * cm.eps
for r in (128, 256, 512):
    cm.RESOLUTION_PARAMS[r]["itermax"] = 500

out = {}
for mfact in [1, 2, 4]:
    res = 512 // mfact
    out[res] = cm.run_reconstruction_for_mfact(mfact)

with open("cache/multiresolution_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("Saved multiresolution cache.")
for res in (512, 256, 128):
    r = out[res]
    pct = (r['final_rmse_single'] - r['final_rmse_two']) / r['final_rmse_single'] * 100
    print(f"  {res}^2: single={r['final_rmse_single']:.4f}, "
          f"two={r['final_rmse_two']:.4f}, gap={pct:+.1f}%")
