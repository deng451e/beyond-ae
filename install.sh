pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
pip install cmake==3.30.4 lm-eval==0.3.0 ftfy accelerate fschat==0.2.36 cvxpy
pip install tqdm rouge jieba fuzzywuzzy einops python-Levenshtein einops Mosek
 
# for mask generation
 
gurobipy==11.0.1


cd $PWD/3rdparty/flashinfer/python
pip install -e .
echo "flashinfer installed"

cd $PWD/3rdparty/Infinigen/speedup
pip install -e infinigen
pip install -e flexgen
echo "Infinigen installed"

cd $PWD/3rdparty/MoA
git checkout 0.0.1
pip install -e .
echo "MoA installed"

cd $PWD/3rdparty/streaming-llm
python setup.py develop
echo "streaming-llm installed"



cd $PWD
python setup.py develop
