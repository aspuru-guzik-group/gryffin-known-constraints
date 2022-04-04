# gryffin-known-constraints

Results for Bayesian optimization with known experimental and design constraints for chemistry applications (https://arxiv.org/abs/2203.17241)


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

## Citation

The results in this repo come from open source research. If you make use of these results in scientific publications, please
cite the following article

```
@misc{hickman_constraints_2022,
      title={Bayesian optimization with known experimental and design constraints for chemistry applications}, 
      author={Riley J. Hickman, Matteo Aldeghi and Florian Häse  and Alán Aspuru-Guzik},
      year={2022},
      eprint={2203.17241},
      archivePrefix={arXiv},
      primaryClass={math.OC}
      }
```
