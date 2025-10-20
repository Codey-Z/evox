# EvoX AI Coding Agent Instructions

## Project Overview
EvoX is a distributed GPU-accelerated evolutionary computation framework built on **PyTorch** (v1.0.0+). It provides 50+ evolutionary algorithms and 100+ benchmark problems for single/multi-objective optimization, neuroevolution, and reinforcement learning tasks.

**Key Architecture**: The codebase follows a modular, functional-first design based on `torch.nn.Module`, enabling `torch.compile` and `torch.vmap` compatibility.

## Core Programming Model

### ModuleBase Hierarchy
All components inherit from `evox.core.ModuleBase` (extends `torch.nn.Module`):
- **Algorithm**: Implements evolutionary strategies (PSO, NSGA-II, CMA-ES, etc.)
- **Problem**: Defines fitness evaluation functions
- **Workflow**: Orchestrates algorithm-problem interaction (e.g., `StdWorkflow`)
- **Monitor**: Tracks optimization progress (e.g., `EvalMonitor`)

### State Management Patterns
```python
# Use Parameter() for static hyperparameters (identified by HPOProblemWrapper)
self.phi = Parameter(0.5, device=device)

# Use Mutable() for values that change during evolution
self.pop = Mutable(torch.rand(pop_size, dim, device=device))

# Access state via .state_dict() / .load_state_dict() for functional workflows
```

### Critical Compilation Rules
1. **Indexing Fix**: Always use `evox.compile()` and `evox.vmap()` instead of `torch.compile()`/`torch.vmap()` - these wrappers fix scalar tensor indexing issues (PyTorch issue #124423)
2. **Control Flow**: Replace Python `if/else` with `torch.cond()` in methods used with `vmap`
3. **External Functions**: Non-compilable code (e.g., Brax simulators) must use:
   - `@torch.compiler.disable` decorator, OR
   - `evox.utils.register_vmap_op()` to register custom operators with vmap support

### Functional Programming with `use_state`
```python
# Convert stateful modules to functional form
stateful_step = use_state(workflow.step)
vmap_stateful_step = vmap(stateful_step, randomness="different")
batch_state = torch.func.stack_module_state([workflow] * 3)
new_batch_state = vmap_stateful_step(batch_state)
```

## Standard Development Workflow

### Creating Algorithms
1. Inherit from `evox.core.Algorithm`
2. Define `__init__()` with `Parameter()` for hyperparameters, `Mutable()` for state
3. Implement `step()` method (calls `self.evaluate(pop)` - injected by workflow)
4. Store results in `self.pop` and `self.fit`
5. Examples: `src/evox/algorithms/so/pso_variants/cso.py`

### Creating Problems
1. Inherit from `evox.core.Problem`
2. Implement `evaluate(pop: torch.Tensor) -> torch.Tensor`
3. For multi-objective, optionally implement `pf()` for Pareto front
4. Examples: `src/evox/problems/numerical/dtlz.py`

### Running Workflows
```python
algorithm = PSO(pop_size=100, lb=lower_bounds, ub=upper_bounds)
problem = Ackley()
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()
```

## Testing Conventions
- Unit tests in `unit_test/` mirror `src/evox/` structure
- Use `torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")`
- Test both compiled and non-compiled workflows
- Example: `unit_test/workflows/test_std_workflow.py`

## Key Utilities
- **ParamsAndVector**: Convert neural network parameters ↔ population vectors for neuroevolution (`src/evox/utils/parameters_and_vector.py`)
- **register_vmap_op**: Make non-PyTorch code (e.g., Brax) compatible with `torch.vmap` (`src/evox/utils/op_register.py`)
- **Operators**: Pre-built crossover/mutation/selection in `src/evox/operators/`

## Device Management
- Set default device: `torch.set_default_device("cuda")` 
- Workflows accept `device` parameter to move all components
- Always use `.to(device)` for bounds/tensors in algorithm `__init__`

## Installation Context
- Python 3.10+ required
- Windows users: `docs/source/_static/win-install.bat` handles CUDA, triton-windows, conda setup
- Dependencies: PyTorch 2.6.0+, optional `[vis]`, `[neuroevolution]`, `[test]` extras (see `pyproject.toml`)

## Documentation
- Main docs: `docs/source/` (Sphinx with MyST)
- Developer guides: `docs/source/guide/developer/`
- Examples: `docs/source/examples/*.ipynb`
- Dual language support: English/Chinese (README.md/README_ZH.md)

## Common Pitfalls
- ❌ Don't use raw `torch.compile` - use `evox.compile`
- ❌ Don't call `.eval()` on ModuleBase - disabled to prevent ambiguity
- ❌ Don't put mutable state in `__init__` - use `Mutable()` wrapper
- ❌ Don't use `if` with Python booleans in vmap contexts - use `torch.cond`
- ✅ Always test with `torch.compile` enabled
- ✅ Use `torch.func.stack_module_state()` for batched workflows
