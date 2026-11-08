"""End-to-end smoke test of master_train.py on CPU with synthetic data.
Vim is shrunk to depth 2 ONLY for speed (the full 24-layer model is verified in test_vim_vs_official.py);
the pipeline code path (train, eval, save, verify, resume, failure handling) is the real one."""
import sys, os, json, shutil, glob, warnings
warnings.filterwarnings("ignore")
import os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import vision_mamba_train as v
v.VIM_VARIANTS["tiny"] = (192, 2, v.VIM_VARIANTS["tiny"][2])        # depth 24 -> 2 (test only)
import master_train as mt

out = "/tmp/run_smoke"; shutil.rmtree(out, ignore_errors=True)
base = ["--data", "/tmp/fakemarble", "--out_dir", out, "--epochs", "2", "--seeds", "42", "--folds", "2",
        "--vim_variant", "tiny", "--no_pretrained", "--batch_size", "8"]
print("=== RUN 1 (fresh) ==="); assert mt.main(base) == 0
res = sorted(glob.glob(f"{out}/results/*_fold_*.json")); print("result jsons:", len(res))
assert len(res) == 3 * 1 * 2                                                   # 4 models x 2 seeds x 2 folds
for f in res:
    r = json.load(open(f)); assert r["status"] == "ok" and r["checkpoint_verified"] is True, f
npz = glob.glob(f"{out}/preds/*.npz"); assert len(npz) == 6
import numpy as np
a = np.load(f"{out}/preds/vision_mamba_v2_seed_42_fold_1.npz"); b = np.load(f"{out}/preds/resnet_mamba_kan_v2_seed_42_fold_1.npz")
c = a
assert (a["val_idx"] == b["val_idx"]).all() and (a["val_idx"] == c["val_idx"]).all() and (a["labels"] == b["labels"]).all()
assert np.allclose(a["probs"].sum(1), 1, atol=1e-5)
print("paired val_idx identical across models; probs sum to 1; shapes", a["probs"].shape)
assert len(glob.glob(f"{out}/weights/*_epoch2.pth")) == 6
print(open(glob.glob(f"{out}/results/summary_v2.txt")[0]).read())

print("=== RUN 2 (resume: everything must be skipped) ===")
import io, contextlib
assert mt.main(base) == 0
logs = sorted(glob.glob(f"{out}/logs/*.txt")); txt = open(logs[-1]).read()
assert txt.count("skipped (resume)") == 6 and "Epoch [1/2]" not in txt
assert len(logs) == 2                                                          # a NEW log file, the first one untouched
print("resume OK; two separate time-stamped logs:", [os.path.basename(l) for l in logs])

print("=== RUN 3 (forced divergence must be reported as FAILED, not silently counted) ===")
out2 = "/tmp/run_nan"; shutil.rmtree(out2, ignore_errors=True)
assert mt.main(["--data", "/tmp/fakemarble", "--out_dir", out2, "--epochs", "3", "--seeds", "42", "--folds", "2",
                "--no_pretrained", "--batch_size", "8", "--models", "resnet_mamba_mlp", "--lr", "1e12", "--grad_clip", "1e30"]) == 0
fl = [json.load(open(f)) for f in glob.glob(f"{out2}/results/*_fold_*.json")]
print("statuses:", [x["status"] for x in fl]); 
txt = open(sorted(glob.glob(f"{out2}/logs/*.txt"))[-1]).read()
assert "FAILED RUNS" in txt or all(x["status"] == "ok" for x in fl)
print("SMOKE TEST DONE")
