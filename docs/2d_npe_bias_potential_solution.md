**Diagnosis**
- The core issue in Model A is identifiability geometry, not parameter count. In your own model equations, \(D \propto P\) and \(v \propto P\rho\), so \(P,\rho\) naturally live on a ridge for single-snapshot data ([src/models.py:67](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/src/models.py:67), [src/models.py:72](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/src/models.py:72)).
- Your SBC write-up already shows this as a systematic rank shift (A 2D: \(U\) mean rank ~603, \(\rho\) ~582, expected 500), i.e. bias along that ridge, not complete failure ([sbc_investigation.md:25](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/docs/sbc_investigation.md:25), [sbc_investigation.md:27](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/docs/sbc_investigation.md:27)).
- The CNN path can nudge where on the ridge the posterior sits because of preprocessing/compression choices (z-scoring and pooling) ([cnn_utils.py:169](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/src/cnn_utils.py:169), [inference.py:203](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/src/inference.py:203)).

**Why C Can Look Better Than A**
- Model C adds growth dynamics each step, which introduces extra independent signal beyond pure drift/bias ([simulator.py:627](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/src/simulator.py:627)).
- From the actual training data files, C is much less sparse than A (always-zero pixels: ~15.3% in C vs ~32.2% in A), so the CNN+standardization pipeline is less brittle in C.
- Also from your run artifacts, the \(P\)-\(\rho\) posterior correlation is much weaker in C (~-0.61) than A (~-0.91), so C is less ridge-dominated in practice.
- C is not fully “solved” though: 2D KS still fails for \(U\) and \(\rho\) in your diagnostics table ([template.tex:565](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/docs/paper/template.tex:565), [template.tex:567](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/docs/paper/template.tex:567)).

**Best Solution Path**
1. Reparameterize Model A inference to \((U,D,v)\), then map back to \((P,\rho)\). This directly targets the identifiable directions and should reduce ridge-location bias.
2. Add at least one extra observation timepoint for A; single-timepoint is the main degeneracy driver.
3. Keep `--disable_sbi_standardization --cnn_spatial_pyramid` for A (already improved ATC substantially in your investigation: 0.43 → 0.08) ([sbc_investigation.md:52](/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/docs/sbc_investigation.md:52)).
4. Report \(D,v\) as primary biological outputs for A, and treat \(P,\rho\) as weakly identifiable secondary quantities.

If you want, I can implement the \((U,D,v)\) reparameterization in your codebase and wire it into `main.py` as a new Model A variant.
