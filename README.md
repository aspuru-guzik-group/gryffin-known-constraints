# gryffin-known-constraints

Results for Bayesian optimization with known experimental and design constraints for chemistry applications


## Installation

## Running a simple example

The file `run_example.py` contains a simple example where the 2-dimensional Dejong surface is optimized with a
constraint on the sum of the parameter values. The constraint is implemented with the following Python function

```python
# constraints
def known_constraints(param):
	''' constraint that the sum of the parameter values must
	less than 1.2
	'''
	param_0_val = param['param_0']
	param_1_val = param['param_1']
	if param_0_val + param_1_val > 1.2:
		return False
	else:
		return True

# initialize gryffin
gryffin = Gryffin(
	config_dict=config,
	known_constraints=known_constraints,
)
```


To run this example, you will first need to install the `Olympus` [package](https://github.com/aspuru-guzik-group/olympus) to access the surafce. You can do this by running

```bash
pip install olymp
python run_example.py
```
