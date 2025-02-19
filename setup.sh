pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
pip install -e .
pip install -r requirements.txt
pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib
git clone git@hf.co:Qwen/Qwen2.5-3B-Instruct
git clone org-14957082@github.com:openai/mle-bench.git
cd mle-bench
sed -i's/>=3\.11/>=3.10/g' pyproject.toml
pip install -e .
cd ..
