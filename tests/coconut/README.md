# COCONUT Test Suite - Running Instructions

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r test-requirements.txt
```

2. **Run the debug test first:**
```bash
pytest test_coconut.py::TestCoconutMainFunction::test_debug_coconut_steps -v -s
```

3. **Run all tests:**
```bash
pytest test_coconut.py -v
```

## 🐛 Debugging Common Issues

### Issue: "TypeError: tuple indices must be integers or slices, not tuple"

This error typically occurs in pandas DataFrame indexing operations. Here's how to debug:

1. **Run the debug test to identify the exact location:**
```bash
pytest test_coconut.py::TestCoconutMainFunction::test_debug_coconut_steps -v -s
```

2. **Check data structure validation:**
```bash
pytest test_coconut.py::TestCoconutMainFunction::test_data_structure_validation -v -s
```

3. **Run the simple test first:**
```bash
pytest test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality -v -s
```

### Common Causes:

1. **Missing imports:** Make sure the COCONUT module functions are properly imported
2. **Index mismatch:** Gene names or sample names don't match between datasets
3. **Empty common genes:** No genes are shared across all datasets
4. **Wrong pandas indexing:** Using tuple indexing where scalar/slice expected

## 📋 Test Structure

### Test Organization:
- **Unit Tests:** Test individual functions in isolation
- **Integration Tests:** Test complete workflows
- **Edge Case Tests:** Test error conditions and boundary cases
- **Performance Tests:** Test with larger datasets

### Test Files:
- `test_coconut.py` - Main test suite
- `pytest.ini` - Test configuration
- `test-requirements.txt` - Test dependencies
- `run_tests.py` - Test runner script

## 🔧 Fixing Import Issues

If you get import errors, you need to ensure the COCONUT module is properly accessible. Here are the options:

### Option 1: Direct import (if COCONUT is in the same directory)
```python
# At the top of test_coconut.py
from coconut_improved import (
    coconut, combine_coco_output, get_a_prior, get_b_prior,
    _prepare_batch_labels, _split_combat_data, _combat_custom,
    _combat_get_params_no_cov, _combat_apply_params_no_cov,
    _it_sol, _int_eprior, BayesParams, GeneExpressionData
)
```

### Option 2: Package installation (recommended)
```bash
# Create a setup.py and install in development mode
pip install -e .
```

### Option 3: PYTHONPATH setup
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_coconut.py -v
```

## 🧪 Running Specific Test Categories

### Run only unit tests:
```bash
pytest test_coconut.py -v -m "not integration and not slow"
```

### Run only integration tests:
```bash
pytest test_coconut.py::TestIntegrationTests -v
```

### Run only fast tests:
```bash
pytest test_coconut.py -v -m "not slow"
```

### Run with coverage:
```bash
pytest test_coconut.py -v --cov=coconut --cov-report=html
```

## 📊 Expected Test Results

When all tests pass, you should see:
- ✅ Data structure validation passes
- ✅ Simple COCONUT functionality works
- ✅ Both parametric and non-parametric modes work
- ✅ Edge cases are handled properly
- ✅ Error conditions raise appropriate exceptions

## 🔍 Debugging Tips

1. **Add print statements:** Use the debug test to see intermediate values
2. **Check data types:** Ensure DataFrames have correct index types
3. **Validate dimensions:** Make sure gene/sample counts match expectations
4. **Test incrementally:** Start with simple tests, then add complexity

## 📁 Required File Structure

```
project/
├── coconut.py                 # Main COCONUT implementation
├── test_coconut.py           # Test suite
├── test_requirements.txt     # Test dependencies
├── pytest.ini              # Pytest configuration
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## 💡 Performance Notes

- **Large dataset tests** are marked as "slow" and can be skipped for quick testing
- **Memory tests** require the `psutil` package
- **Coverage reports** require the `pytest-cov` package

## 📞 Getting Help

If tests continue to fail:
1. Check that all required packages are installed
2. Verify the COCONUT module can be imported
3. Run the debug test to identify the specific failure point
4. Check that test data has the expected structure