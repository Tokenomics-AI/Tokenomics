# Diagnostic Tests

This directory contains comprehensive diagnostic tests that validate all platform components.

## Tests

### `extensive_diagnostic_test.py`

Comprehensive 32-query test across 10 phases that triggers every platform component:
- Exact cache hits
- Semantic cache hits (direct + context injection)
- Complexity analysis
- Strategy selection
- Query compression
- Quality judging

**Usage:**
```bash
python tests/diagnostic/extensive_diagnostic_test.py
```

**Results:** Saved to `tests/diagnostic/results/`

### `quick_fix_validation.py`

Quick validation test for specific fixes:
- Complexity classification
- Strategy selection
- Tone detection
- Query compression

**Usage:**
```bash
python tests/diagnostic/quick_fix_validation.py
```

### `test_setup.py`

Basic setup verification test.

**Usage:**
```bash
python tests/diagnostic/test_setup.py
```

## Results

Test results are stored in `results/` subdirectory:
- `extensive_diagnostic_results.json` - Full diagnostic test results
- `DIAGNOSTIC_ISSUES_REPORT.md` - Issues identified
- `PERFORMANCE_IMPACT_ANALYSIS.md` - Performance improvements
- `FIX_VALIDATION_RESULTS.md` - Fix validation results

## Test Evolution

See [docs/testing/test-evolution.md](../../docs/testing/test-evolution.md) for the complete story of how these tests evolved and improved the platform.









