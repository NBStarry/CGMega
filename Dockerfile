FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update || true

RUN apt-get install -y graphviz libgraphviz-dev

WORKDIR /CGMega

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install jupyter

RUN pip install torch_scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

RUN pip install torch_sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

RUN pip install torch_geometric==2.0.3 transformers wandb matplotlib pygraphviz

COPY ./ /CGMega/

EXPOSE 8888

CMD ["bash"]