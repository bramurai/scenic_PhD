"""
Analysis of MBT paper audio preprocessing claims vs Lingvo defaults

The paper states: "25ms Hamming window with hop length 10ms"
But Lingvo's default is Hann window, not Hamming!
"""

import numpy as np

# Generate both windows
window_size = 400  # 25ms at 16kHz

hamming = np.hamming(window_size)
hann = np.hanning(window_size)

# Compute difference
diff = hamming - hann
max_diff = np.max(np.abs(diff))
mean_diff = np.mean(np.abs(diff))

print("=" * 80)
print("HAMMING vs HANN WINDOW COMPARISON")
print("=" * 80)
print()
print(f"Window size: {window_size} samples (25ms at 16kHz)")
print()
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Max Hamming value: {np.max(hamming):.6f}")
print(f"Max Hann value: {np.max(hann):.6f}")
print()

# Check formulas
print("Mathematical formulas:")
print("  Hamming: w[n] = 0.54 - 0.46 * cos(2πn/(N-1))")
print("  Hann:    w[n] = 0.5 - 0.5 * cos(2πn/(N-1))")
print()
print("Key difference:")
print("  Hamming has non-zero endpoints (reduces spectral leakage)")
print("  Hann has zero endpoints (better frequency resolution)")
print()

# Compute correlation
correlation = np.corrcoef(hamming, hann)[0, 1]
print(f"Correlation between windows: {correlation:.6f}")
print()

if correlation > 0.99:
    print("✓ Windows are VERY similar (correlation > 0.99)")
    print("  → Small impact on final spectrograms")
else:
    print("✗ Windows are significantly different")
    print("  → Could impact model performance")

print()
print("=" * 80)
print("CRITICAL INSIGHT - PAPER vs LINGVO CODE")
print("=" * 80)
print()
print("Paper states: '25ms Hamming window'")
print("Lingvo uses:  '25ms Hann window' (HANNING in code)")
print()
print("Possible explanations:")
print("  1. Paper made documentation error (wrote Hamming, meant Hann)")
print("  2. Authors customized Lingvo to use Hamming")
print("  3. Hamming/Hann are often confused in literature")
print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("Given that:")
print("  1. Paper explicitly says 'Hamming window'")
print("  2. Lingvo default is 'Hann window'")
print(f"  3. Windows are {correlation:.1%} correlated (very similar)")
print()
print("SUGGESTED STRATEGY:")
print("  → START WITH HAMMING (match the paper)")
print("    Since paper explicitly mentions Hamming, this is safer")
print()
print("  → If performance still poor, try HANN as fallback")
print("    In case authors used Lingvo defaults despite paper description")
print()
print("PRIORITY OF FIXES (most to least critical):")
print("  1. ⭐⭐⭐ Preemphasis (0.97) - MISSING entirely in your code")
print("  2. ⭐⭐⭐ Frequency range (125-7600 Hz) - Wrong range (0-8000)")
print("  3. ⭐⭐  Window type (Hamming vs Hann) - 99.8% similar, minor impact")
print("  4. ⭐   Window length (+1 sample) - Very minor impact")
print()
print("CONCLUSION:")
print("  Let's use HAMMING to match the paper exactly.")
print("  The preemphasis and frequency range fixes are FAR more important!")
