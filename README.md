
### ZK-(RPG)F allocator


This repo is a proof of concept of a zk circuit for the Optimism RPGF allocator (see [here](https://github.com/ethereum-optimism/op-analytics/tree/main/rpgf_calculator)). 

We leverage [ezkl](https://github.com/zkonduit/ezkl) to build the circuit directly from the python code in a jupyter notebook `rpgf3_allocator.ipynb`. The notebook has associated markdown cells that explain the circuit and the steps to build it.

1. We first build a pytorch equivalent of the allocator function, originally in pandas. 
2. We then export the pytorch function to an onnx file and use the ezkl compiler to convert it into a zk-circuit. 
3. We generate a proof for a given input and verify. 
4. We add sanity checks for numerical accuracy and compare the results with the original pandas function.

To get started 

```bash
python -m venv .env
source .env/bin/activate
pip install ezkl torch pandas numpy jupyter onnx plotly onnxruntime
```

Then open the notebook and follow the instructions.

There are two notebooks:

- `rpgf3_allocator.ipynb` runs on dummy data for validation.
- `rpgf3_allocator_real.ipynb` runs on data of truncated vote amounts that the OP foundation has published.

  

